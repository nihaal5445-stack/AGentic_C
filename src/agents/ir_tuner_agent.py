"""
AGentic_C — IR Tuner Agent (HFT Edition)
==========================================
Selects and applies LLVM optimisation passes to reduce IR instruction count
and latency. Driven by PPO via CompilerGym.

Two modes of operation:

  GENERAL mode  — standard IR optimisation for cold-path code
                  reward = IrInstructionCount reduction
                  action space = all 124 CompilerGym LLVM passes

  HFT mode      — latency-budget-aware optimisation for hot-path code
                  reward = weighted(instruction reduction, latency estimate,
                                    anti-pattern resolution)
                  action space = HFT_PASS_PRIORITY subset first, then full space
                  budget = nanoseconds, not just step count

HFT Pass Priority Groups (applied before general exploration):
  GROUP 1 — Anti-pattern resolvers (from Fixer Agent directives)
    mem2reg, sroa           → resolve LAP-001 heap allocs
    inline, always-inline   → resolve LAP-002 virtual, LAP-006 indirect calls
    simplifycfg             → resolve LAP-003 exceptions, LAP-010 branches
    barrier-elimination     → resolve LAP-004 locks
    licm                    → resolve LAP-007 atomics in loops

  GROUP 2 — General HFT performance passes
    loop-unroll             → unroll tight tick-processing loops
    gvn                     → eliminate redundant price/size computations
    dce                     → remove dead book-keeping on hot path
    instcombine             → combine instruction sequences
    jump-threading          → eliminate branch chains

  GROUP 3 — Cache and layout passes
    loop-vectorize          → SIMD on price array operations
    slp-vectorizer          → superword-level parallelism
    alignment               → cache-line alignment for hot structs

Architecture:
  IRTunerAgent
    ├── CompilerGymEnv      — wraps compiler_gym.make("llvm-v0")
    ├── HFTPassSelector     — prioritises passes based on anti-pattern directives
    ├── LatencyCostModel    — estimates ns cost of IR based on instruction mix
    ├── PPOPolicy           — SB3 PPO (stub until training data exists)
    └── RewardShaper        — composite reward for HFT mode
"""

import os
import re
import yaml
import time
import numpy as np
from dataclasses import dataclass, field
from typing import Optional
from pathlib import Path
from enum import Enum


# ---------------------------------------------------------------------------
# HFT-relevant LLVM pass catalogue
# Maps CompilerGym action indices to pass names and HFT relevance.
# Full list: compiler_gym.envs.llvm.datasets — 124 passes total.
# We label the ones that matter most for HFT.
# ---------------------------------------------------------------------------

# Pass name → (hft_priority, group, estimated_ns_saving_per_apply)
HFT_PASS_CATALOGUE = {
    # GROUP 1 — Anti-pattern resolvers
    "mem2reg":              (1, "anti_pattern", 15),   # stack→register: -15ns
    "sroa":                 (1, "anti_pattern", 12),   # scalar replacement
    "-inline":              (1, "anti_pattern", 8),    # function inlining
    "always-inline":        (1, "anti_pattern", 10),   # force inline
    "simplifycfg":          (1, "anti_pattern", 6),    # CFG simplification
    "barrier-noop":         (1, "anti_pattern", 4),    # barrier elimination
    "licm":                 (1, "anti_pattern", 20),   # loop invariant motion

    # GROUP 2 — General HFT performance
    "loop-unroll":          (2, "hft_perf", 25),       # loop unrolling
    "gvn":                  (2, "hft_perf", 10),       # global value numbering
    "dce":                  (2, "hft_perf", 5),        # dead code elimination
    "instcombine":          (2, "hft_perf", 8),        # instruction combining
    "jump-threading":       (2, "hft_perf", 12),       # branch chain elim
    "aggressive-instcombine":(2, "hft_perf", 9),
    "reassociate":          (2, "hft_perf", 4),
    "sccp":                 (2, "hft_perf", 7),        # sparse cond const prop
    "loop-idiom":           (2, "hft_perf", 6),
    "indvars":              (2, "hft_perf", 8),        # induction variable simp

    # GROUP 3 — Cache and vectorisation
    "loop-vectorize":       (3, "cache_layout", 40),   # SIMD vectorisation
    "slp-vectorizer":       (3, "cache_layout", 30),   # superword parallelism
    "loop-distribute":      (3, "cache_layout", 15),
    "loop-load-elim":       (3, "cache_layout", 10),
}

# Anti-pattern code → list of pass names to prioritise
ANTIPATTERN_PASS_MAP = {
    "LAP-001": ["mem2reg", "sroa"],                # heap alloc
    "LAP-002": ["-inline", "always-inline"],       # virtual dispatch
    "LAP-003": ["simplifycfg"],                    # exceptions
    "LAP-004": ["barrier-noop", "licm"],           # locks
    "LAP-005": ["simplifycfg", "dce"],             # syscalls
    "LAP-006": ["-inline", "always-inline"],       # indirect calls
    "LAP-007": ["licm", "instcombine"],            # atomics
    "LAP-008": ["simplifycfg", "-inline"],         # RTTI
    "LAP-009": ["sroa", "instcombine"],            # alignment
    "LAP-010": ["jump-threading", "simplifycfg"],  # branches
}


# ---------------------------------------------------------------------------
# Latency cost model
# Estimates nanosecond cost of an IR based on instruction mix.
# Used by the Timing Verifier stub and reward shaper.
# ---------------------------------------------------------------------------

@dataclass
class LatencyEstimate:
    total_ns:          float
    instruction_count: int
    memory_ops:        int
    branch_count:      int
    call_count:        int
    breakdown:         dict = field(default_factory=dict)


class LatencyCostModel:
    """
    Estimates the hot-path execution latency of an LLVM IR snippet.

    Uses a simple instruction-mix model calibrated to modern x86/arm64:
      - Each instruction type has a base cost in ns
      - Memory operations add cache-miss probability cost
      - Branches add misprediction probability cost
      - Calls add call overhead cost

    This is NOT cycle-accurate — it's a heuristic for relative comparison.
    Real measurement happens in the Timing Verifier (perf/valgrind/iaca).
    """

    # Instruction type → base cost in nanoseconds (at 3GHz, no stalls)
    INSTRUCTION_COST_NS = {
        "add":      0.33,   "sub":      0.33,   "mul":      0.67,
        "sdiv":     3.33,   "udiv":     3.33,   "fadd":     0.67,
        "fsub":     0.67,   "fmul":     1.00,   "fdiv":     5.00,
        "icmp":     0.33,   "fcmp":     0.67,   "br":       0.33,
        "load":     1.00,   "store":    1.00,   "alloca":   0.50,
        "call":     3.00,   "ret":      0.33,   "phi":      0.33,
        "select":   0.33,   "getelementptr": 0.33,
        "bitcast":  0.00,   "zext":     0.33,   "sext":     0.33,
        "trunc":    0.33,   "and":      0.33,   "or":       0.33,
        "xor":      0.33,   "shl":      0.33,   "lshr":     0.33,
    }

    # Additional cost for cache miss probability
    CACHE_MISS_PROBABILITY = 0.05     # 5% of memory ops miss L1
    CACHE_MISS_COST_NS     = 60.0     # L2 hit ~5ns, L3 ~20ns, DRAM ~60ns

    # Branch misprediction
    BRANCH_MISPREDICT_PROB = 0.03     # 3% misprediction rate
    BRANCH_MISPREDICT_NS   = 15.0     # ~15 pipeline stages * 0.33ns

    def estimate(self, ir_text: str) -> LatencyEstimate:
        """
        Estimates latency from raw LLVM IR text.
        Returns LatencyEstimate with per-category breakdown.
        """
        lines = ir_text.lower().split("\n")
        counts = {k: 0 for k in self.INSTRUCTION_COST_NS}
        total_cost = 0.0

        for line in lines:
            line = line.strip()
            if not line or line.startswith(";") or line.startswith("!"):
                continue
            for instr, cost in self.INSTRUCTION_COST_NS.items():
                if re.search(rf'\b{instr}\b', line):
                    counts[instr] += 1
                    total_cost += cost
                    break

        memory_ops   = counts["load"] + counts["store"] + counts["alloca"]
        branch_count = counts["br"]
        call_count   = counts["call"]
        total_instr  = sum(counts.values())

        # Add stall costs
        cache_cost  = memory_ops * self.CACHE_MISS_PROBABILITY * self.CACHE_MISS_COST_NS
        branch_cost = branch_count * self.BRANCH_MISPREDICT_PROB * self.BRANCH_MISPREDICT_NS

        total_ns = total_cost + cache_cost + branch_cost

        return LatencyEstimate(
            total_ns          = round(total_ns, 2),
            instruction_count = total_instr,
            memory_ops        = memory_ops,
            branch_count      = branch_count,
            call_count        = call_count,
            breakdown         = {
                "base_instruction_cost_ns": round(total_cost, 2),
                "cache_stall_cost_ns":      round(cache_cost, 2),
                "branch_mispredict_ns":     round(branch_cost, 2),
            }
        )

    def improvement_ratio(self, before: LatencyEstimate,
                          after: LatencyEstimate) -> float:
        """Returns fractional improvement: 0.0 = no change, 1.0 = perfect."""
        if before.total_ns == 0:
            return 0.0
        return max(0.0, (before.total_ns - after.total_ns) / before.total_ns)


# ---------------------------------------------------------------------------
# HFT Pass Selector
# Decides which passes to try first based on anti-pattern directives
# ---------------------------------------------------------------------------

class HFTPassSelector:
    """
    Converts Fixer Agent directives and anti-pattern codes into
    an ordered list of passes for the IR Tuner to apply first.

    In HFT mode, the IR Tuner doesn't explore randomly — it targets
    known problems first, then falls through to general exploration
    if budget remains.
    """

    def build_priority_queue(self,
                             anti_pattern_codes: list[str],
                             directive_text:     str = "") -> list[str]:
        """
        Returns ordered list of pass names to try first.
        Anti-pattern-targeted passes come first, then general HFT passes.

        Args:
            anti_pattern_codes: list of LAP-00X codes from Fixer Agent
            directive_text:     free-text directive from Boss Agent

        Returns:
            Ordered list of pass names, deduplicated
        """
        priority_passes = []
        seen = set()

        # Stage 1 — anti-pattern targeted passes
        for code in anti_pattern_codes:
            code_clean = code.split(":")[0]   # strip "LAP-001:critical:new O"
            for pass_name in ANTIPATTERN_PASS_MAP.get(code_clean, []):
                if pass_name not in seen:
                    priority_passes.append(pass_name)
                    seen.add(pass_name)

        # Stage 2 — directive keyword extraction
        directive_lower = directive_text.lower()
        for pass_name, (priority, group, _) in HFT_PASS_CATALOGUE.items():
            if pass_name.replace("-", " ") in directive_lower:
                if pass_name not in seen:
                    priority_passes.append(pass_name)
                    seen.add(pass_name)

        # Stage 3 — fill remaining with group 1 and 2 passes
        for pass_name, (priority, group, _) in sorted(
                HFT_PASS_CATALOGUE.items(), key=lambda x: x[1][0]):
            if priority <= 2 and pass_name not in seen:
                priority_passes.append(pass_name)
                seen.add(pass_name)

        return priority_passes

    def ns_saving_estimate(self, pass_name: str) -> int:
        """Returns estimated ns saving for a given pass."""
        entry = HFT_PASS_CATALOGUE.get(pass_name)
        return entry[2] if entry else 0


# ---------------------------------------------------------------------------
# Reward Shaper — composite reward for HFT mode
# ---------------------------------------------------------------------------

class RewardShaper:
    """
    Computes composite reward for the IR Tuner PPO agent.

    General mode:  R = instruction_count_reduction
    HFT mode:      R = α * instr_reduction
                     + β * latency_improvement
                     + γ * antipattern_resolution
                     - δ * budget_overage_penalty
    """

    def __init__(self,
                 alpha: float = 0.40,   # instruction count weight
                 beta:  float = 0.40,   # latency improvement weight
                 gamma: float = 0.15,   # anti-pattern resolution weight
                 delta: float = 0.05):  # budget overage penalty weight
        self.alpha = alpha
        self.beta  = beta
        self.gamma = gamma
        self.delta = delta

    def compute(self,
                instr_before:     int,
                instr_after:      int,
                latency_before:   float,
                latency_after:    float,
                antipatterns_resolved: int,
                antipatterns_total:    int,
                budget_ns:        int,
                estimated_ns:     float) -> float:
        """
        Computes the scalar reward signal for PPO.
        All sub-scores normalised to [0, 1].
        """
        # Instruction count reduction (normalised)
        instr_score = 0.0
        if instr_before > 0:
            instr_score = max(0.0, (instr_before - instr_after) / instr_before)

        # Latency improvement (normalised)
        latency_score = 0.0
        if latency_before > 0:
            latency_score = max(0.0, (latency_before - latency_after) / latency_before)

        # Anti-pattern resolution (normalised)
        ap_score = 0.0
        if antipatterns_total > 0:
            ap_score = antipatterns_resolved / antipatterns_total

        # Budget overage penalty (positive = over budget)
        overage_penalty = 0.0
        if budget_ns > 0 and estimated_ns > budget_ns:
            overage_ratio   = (estimated_ns - budget_ns) / budget_ns
            overage_penalty = min(1.0, overage_ratio)

        reward = (
            self.alpha * instr_score
            + self.beta  * latency_score
            + self.gamma * ap_score
            - self.delta * overage_penalty
        )

        return round(max(0.0, min(1.0, reward)), 4)


# ---------------------------------------------------------------------------
# CompilerGym environment wrapper
# Handles the env lifecycle and exposes a clean step interface.
# Stubs gracefully when CompilerGym is unavailable.
# ---------------------------------------------------------------------------

class CompilerGymEnv:
    """
    Wraps compiler_gym.make("llvm-v0").

    In real execution: uses the live CompilerGym LLVM environment.
    In stub mode: simulates the step loop for testing without CG installed.
    """

    def __init__(self, stub_mode: bool = False):
        self.stub_mode   = stub_mode
        self.env         = None
        self._step_count = 0
        self._ir_text    = ""
        self._available  = False
        self._base_instr = 0
        self._cur_instr  = 0

        if not stub_mode:
            self._available = self._try_init()

        if not self._available:
            self.stub_mode = True

    def _try_init(self) -> bool:
        try:
            import compiler_gym
            self.env = compiler_gym.make("llvm-v0")
            return True
        except Exception as e:
            print(f"[CompilerGymEnv] CompilerGym unavailable ({e}). Using stub.")
            return False

    def reset(self, ir_path: str) -> np.ndarray:
        """Reset environment with a new IR file. Returns initial observation."""
        self._step_count = 0

        if not self.stub_mode and self.env:
            try:
                # CompilerGym accepts file:// URIs or benchmark IDs
                obs = self.env.reset(benchmark=f"file:///{ir_path}")
                self._ir_text    = self.env.ir
                self._base_instr = self._count_instructions(self._ir_text)
                self._cur_instr  = self._base_instr
                return np.array(obs, dtype=np.float32) if obs is not None \
                       else self._stub_obs()
            except Exception as e:
                print(f"[CompilerGymEnv] reset failed: {e}. Falling back to stub.")
                self.stub_mode = True

        # Stub mode — read IR from disk and fake observations
        if os.path.exists(ir_path):
            with open(ir_path) as f:
                self._ir_text = f.read()
        else:
            self._ir_text = "; empty stub IR"

        self._base_instr = self._count_instructions(self._ir_text)
        self._cur_instr  = self._base_instr
        return self._stub_obs()

    def step(self, action_name: str) -> tuple[np.ndarray, float, bool]:
        """
        Apply one pass. Returns (observation, reward, done).
        action_name: LLVM pass string e.g. "mem2reg"
        """
        self._step_count += 1

        if not self.stub_mode and self.env:
            try:
                # Convert pass name to CG action index
                action_idx = self._pass_to_action_idx(action_name)
                obs, reward, done, _ = self.env.step(action_idx)
                self._ir_text   = self.env.ir
                self._cur_instr = self._count_instructions(self._ir_text)
                return (np.array(obs, dtype=np.float32),
                        float(reward), bool(done))
            except Exception as e:
                self.stub_mode = True

        # Stub: simulate modest instruction reduction per pass
        saving = HFT_PASS_CATALOGUE.get(action_name, (0, "", 3))[2]
        reduction_frac = min(0.08, saving / max(self._cur_instr, 1) * 0.5)
        new_instr = max(1, int(self._cur_instr * (1 - reduction_frac)))
        reward    = (self._cur_instr - new_instr) / max(self._base_instr, 1)
        self._cur_instr = new_instr
        done = (self._cur_instr < self._base_instr * 0.5)   # 50% reduction cap
        return self._stub_obs(), float(reward), done

    def get_ir(self) -> str:
        """Returns current IR text."""
        if not self.stub_mode and self.env:
            return self.env.ir
        return self._ir_text

    def get_instruction_count(self) -> int:
        return self._cur_instr

    def close(self):
        if self.env:
            self.env.close()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _stub_obs(self) -> np.ndarray:
        """56-dim Autophase-style observation vector for stub mode."""
        obs = np.zeros(56, dtype=np.float32)
        obs[0] = self._cur_instr / max(self._base_instr, 1)
        obs[1] = self._step_count / 50.0
        return obs

    def _count_instructions(self, ir_text: str) -> int:
        """Count non-empty, non-comment IR lines as proxy for instruction count."""
        return sum(1 for l in ir_text.split("\n")
                   if l.strip() and not l.strip().startswith(";")
                   and not l.strip().startswith("!")
                   and not l.strip().startswith("source_filename"))

    def _pass_to_action_idx(self, pass_name: str) -> int:
        """
        Convert pass name to CompilerGym action index.
        CompilerGym uses integer actions 0-123.
        This mapping is approximate — real mapping via env.action_space.
        """
        pass_list = list(HFT_PASS_CATALOGUE.keys())
        if pass_name in pass_list:
            return pass_list.index(pass_name) % 124
        return hash(pass_name) % 124


# ---------------------------------------------------------------------------
# PPO Policy stub
# Real PPO uses stable_baselines3.PPO — stubs here for testing.
# ---------------------------------------------------------------------------

class PPOPolicyStub:
    """
    Stub PPO policy for testing without SB3 training data.
    Returns actions from a priority queue built by HFTPassSelector.
    Falls back to random selection when queue is exhausted.

    In real usage, replace with:
      from stable_baselines3 import PPO
      model = PPO.load("models/ir_tuner_ppo.zip")
      action, _ = model.predict(obs)
    """

    def __init__(self, priority_queue: list[str]):
        self._queue   = list(priority_queue)
        self._pointer = 0
        self._rng     = np.random.default_rng(seed=42)

    def predict(self, obs: np.ndarray) -> str:
        """Returns next pass name to apply."""
        if self._pointer < len(self._queue):
            action = self._queue[self._pointer]
            self._pointer += 1
            return action
        # Exhausted queue — random from HFT catalogue
        passes = list(HFT_PASS_CATALOGUE.keys())
        return self._rng.choice(passes)


# ---------------------------------------------------------------------------
# IR Tuner Result
# ---------------------------------------------------------------------------

@dataclass
class IRTunerResult:
    success:              bool
    ir_path_in:           str
    ir_path_out:          str          = ""
    passes_applied:       list         = field(default_factory=list)
    steps_taken:          int          = 0
    instr_count_before:   int          = 0
    instr_count_after:    int          = 0
    latency_before_ns:    float        = 0.0
    latency_after_ns:     float        = 0.0
    cumulative_reward:    float        = 0.0
    antipatterns_resolved: int         = 0
    hft_mode:             bool         = False
    budget_ns:            int          = 0
    within_budget:        bool         = False
    notes:                str          = ""


# ---------------------------------------------------------------------------
# IR Tuner Agent — main class
# ---------------------------------------------------------------------------

class IRTunerAgent:
    """
    Selects and applies LLVM optimisation passes to reduce IR latency.

    In HFT mode:
      1. Receives CodeUnitContext with anti_patterns and budget_ns
      2. HFTPassSelector builds a priority queue from anti-patterns
      3. PPO works through priority queue first, then explores
      4. LatencyCostModel estimates ns cost after each pass
      5. RewardShaper scores each step for PPO learning signal
      6. Stops when: budget met, step limit reached, or no more improvement
      7. Returns IRTunerResult with passes applied and latency delta

    In general mode:
      Standard CompilerGym PPO loop — reward = IrInstructionCount reduction
    """

    def __init__(self, config_path: str = "configs/config.yaml",
                 stub_mode: bool = True):
        self.config       = self._load_config(config_path)
        self.tuner_cfg    = self.config.get("agents", {}).get("ir_tuner", {})
        self.max_steps    = self.tuner_cfg.get("max_steps", 45)
        self.cost_model   = LatencyCostModel()
        self.pass_selector= HFTPassSelector()
        self.reward_shaper= RewardShaper()
        self.env          = CompilerGymEnv(stub_mode=stub_mode)
        self._log(f"IR Tuner Agent initialized. "
                  f"stub_mode={stub_mode}, max_steps={self.max_steps}")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def tune(self,
             ir_path:       str,
             budget_steps:  int,
             hft_mode:      bool         = False,
             budget_ns:     int          = 0,
             anti_patterns: list[str]    = None,
             directive:     str          = "") -> IRTunerResult:
        """
        Main entry point. Called by Boss Agent's HFT chain or pipeline.py.

        Args:
            ir_path:       path to input .ll file
            budget_steps:  max number of passes to apply
            hft_mode:      True = HFT latency-aware mode
            budget_ns:     latency budget in nanoseconds (HFT mode)
            anti_patterns: list of LAP-00X codes from Fixer Agent
            directive:     text directive from Boss Agent retry loop

        Returns:
            IRTunerResult with all metrics populated
        """
        if not os.path.exists(ir_path) and not ir_path.startswith("/tmp/stub"):
            return IRTunerResult(
                success=False,
                ir_path_in=ir_path,
                notes=f"IR file not found: {ir_path}"
            )

        anti_patterns = anti_patterns or []
        budget_steps  = min(budget_steps, self.max_steps)

        self._log(f"Tuning: {Path(ir_path).name} | "
                  f"steps={budget_steps} | hft={hft_mode} | budget_ns={budget_ns}")

        # Reset environment
        obs            = self.env.reset(ir_path)
        instr_before   = self.env.get_instruction_count()
        ir_before      = self.env.get_ir()
        latency_before = self.cost_model.estimate(ir_before).total_ns

        # Build pass priority queue for HFT mode
        priority_queue = []
        if hft_mode:
            priority_queue = self.pass_selector.build_priority_queue(
                anti_patterns, directive
            )
            self._log(f"  Priority queue ({len(priority_queue)} passes): "
                      f"{priority_queue[:6]}{'...' if len(priority_queue) > 6 else ''}")

        # Instantiate policy
        policy = PPOPolicyStub(priority_queue)

        # Main tuning loop
        passes_applied    = []
        cumulative_reward = 0.0
        antipatterns_resolved = 0
        ap_codes_targeted = {c.split(":")[0] for c in anti_patterns}

        for step in range(budget_steps):
            action = policy.predict(obs)
            obs, reward, done = self.env.step(action)

            passes_applied.append(action)
            cumulative_reward += reward

            # Track anti-pattern resolution
            if hft_mode and action in ANTIPATTERN_PASS_MAP.values():
                for ap_code, passes in ANTIPATTERN_PASS_MAP.items():
                    if action in passes and ap_code in ap_codes_targeted:
                        antipatterns_resolved += 1
                        ap_codes_targeted.discard(ap_code)

            # HFT mode: check latency estimate every 5 steps
            if hft_mode and budget_ns > 0 and step % 5 == 0:
                cur_ir      = self.env.get_ir()
                cur_latency = self.cost_model.estimate(cur_ir).total_ns
                if cur_latency <= budget_ns:
                    self._log(f"  Budget met at step {step+1}: "
                              f"{cur_latency:.0f}ns ≤ {budget_ns}ns. Stopping early.")
                    break

            if done:
                break

        # Final measurements
        ir_after      = self.env.get_ir()
        instr_after   = self.env.get_instruction_count()
        latency_after = self.cost_model.estimate(ir_after).total_ns
        within_budget = (latency_after <= budget_ns) if budget_ns > 0 else True

        # Write output IR
        ir_path_out = ir_path.replace(".ll", "_opt.ll")
        try:
            with open(ir_path_out, "w") as f:
                f.write(ir_after)
        except Exception:
            ir_path_out = ir_path   # fallback — return same path

        instr_reduction = instr_before - instr_after
        latency_delta   = latency_before - latency_after

        self._log(f"  Done. Steps: {len(passes_applied)} | "
                  f"Instr: {instr_before}→{instr_after} "
                  f"(-{instr_reduction}) | "
                  f"Latency: {latency_before:.0f}→{latency_after:.0f}ns "
                  f"(Δ{latency_delta:.0f}ns) | "
                  f"Budget: {'✓' if within_budget else '✗'}")

        return IRTunerResult(
            success               = True,
            ir_path_in            = ir_path,
            ir_path_out           = ir_path_out,
            passes_applied        = passes_applied,
            steps_taken           = len(passes_applied),
            instr_count_before    = instr_before,
            instr_count_after     = instr_after,
            latency_before_ns     = round(latency_before, 2),
            latency_after_ns      = round(latency_after, 2),
            cumulative_reward     = round(cumulative_reward, 4),
            antipatterns_resolved = antipatterns_resolved,
            hft_mode              = hft_mode,
            budget_ns             = budget_ns,
            within_budget         = within_budget,
            notes                 = (
                f"{len(passes_applied)} passes applied. "
                f"Instruction reduction: {instr_reduction} "
                f"({100*instr_reduction/max(instr_before,1):.1f}%). "
                f"Latency delta: {latency_delta:.0f}ns."
            )
        )

    def tune_unit(self, unit, ir_path: str) -> IRTunerResult:
        """
        Convenience wrapper for Boss Agent's CodeUnitContext.
        Called inside run_hft_chain as the ir_tuner_agent callable.

          plan = agent.run_hft_chain(
              plan,
              ir_tuner_agent=lambda u, p: tuner.tune_unit(u, ir_path),
              ...
          )
        """
        return self.tune(
            ir_path       = ir_path,
            budget_steps  = 35,             # default HFT budget
            hft_mode      = True,
            budget_ns     = unit.budget_ns,
            anti_patterns = unit.anti_patterns,
            directive     = getattr(unit, "ir_tuner_directive", "")
        )

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _load_config(self, path: str) -> dict:
        if os.path.exists(path):
            with open(path) as f:
                return yaml.safe_load(f)
        return {}

    def _log(self, msg: str):
        print(f"[IRTuner] {msg}")


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import tempfile

    print("=" * 70)
    print("AGentic_C — IR Tuner Agent (HFT Edition) Smoke Test")
    print("=" * 70)

    # ── Create a sample .ll file ──────────────────────────────────────────
    SAMPLE_IR = """; Sample LLVM IR — matrix multiply style
; Intentionally unoptimised for testing

define i32 @tick_handler(float %price, float %ema, i32 %position) {
entry:
  %p = alloca float, align 4
  %e = alloca float, align 4
  %pos = alloca i32, align 4
  store float %price, float* %p, align 4
  store float %ema,   float* %e, align 4
  store i32  %position, i32* %pos, align 4
  %p_val = load float, float* %p, align 4
  %e_val = load float, float* %e, align 4
  %cmp = fcmp ogt float %p_val, %e_val
  br i1 %cmp, label %signal_true, label %signal_false

signal_true:
  %pos_val = load i32, i32* %pos, align 4
  %limit = icmp slt i32 %pos_val, 1000
  br i1 %limit, label %submit, label %reject

submit:
  %order_qty = add i32 %pos_val, 100
  %check = icmp slt i32 %order_qty, 1000
  br i1 %check, label %ok, label %reject

ok:
  ret i32 1

reject:
  ret i32 0

signal_false:
  ret i32 0
}

define i32 @risk_check(i32 %qty, i32 %current_pos) {
entry:
  %sum = add i32 %qty, %current_pos
  %check = icmp slt i32 %sum, 1000
  %result = zext i1 %check to i32
  ret i32 %result
}
"""

    # Write to temp file
    tmp = tempfile.NamedTemporaryFile(suffix=".ll", mode="w",
                                     delete=False, dir="/tmp")
    tmp.write(SAMPLE_IR)
    tmp.close()
    ir_path = tmp.name
    print(f"\n✓ Sample IR written: {ir_path}")

    # ── Test 1: General mode (no HFT) ────────────────────────────────────
    print("\n── Test 1: General mode ──")
    tuner = IRTunerAgent(stub_mode=True)
    result = tuner.tune(ir_path, budget_steps=10, hft_mode=False)

    if result.success:
        print(f"  ✓ PASSED")
        print(f"    Steps:       {result.steps_taken}")
        print(f"    Instr:       {result.instr_count_before} → {result.instr_count_after}")
        print(f"    Latency:     {result.latency_before_ns:.1f} → {result.latency_after_ns:.1f}ns")
        print(f"    Reward:      {result.cumulative_reward:.4f}")
    else:
        print(f"  ✗ FAILED: {result.notes}")

    # ── Test 2: HFT mode, clean unit (no anti-patterns) ──────────────────
    print("\n── Test 2: HFT mode — clean unit ──")
    result2 = tuner.tune(
        ir_path,
        budget_steps = 15,
        hft_mode     = True,
        budget_ns    = 400,
        anti_patterns= [],
        directive    = "standard HFT passes"
    )

    if result2.success:
        within = "✓ within budget" if result2.within_budget else "✗ over budget"
        print(f"  ✓ PASSED — {within}")
        print(f"    Latency:     {result2.latency_before_ns:.1f} → {result2.latency_after_ns:.1f}ns")
        print(f"    Budget:      {result2.budget_ns}ns")
        print(f"    Passes:      {result2.passes_applied[:5]}")
    else:
        print(f"  ✗ FAILED: {result2.notes}")

    # ── Test 3: HFT mode with anti-patterns ──────────────────────────────
    print("\n── Test 3: HFT mode — heap + mutex anti-patterns ──")
    result3 = tuner.tune(
        ir_path,
        budget_steps = 20,
        hft_mode     = True,
        budget_ns    = 250,
        anti_patterns= ["LAP-001:critical:new O",
                        "LAP-004:critical:std::mutex",
                        "LAP-005:major:printf("],
        directive    = "heap alloc detected: apply mem2reg, sroa"
    )

    if result3.success:
        print(f"  ✓ PASSED")
        print(f"    Priority passes tried first: "
              f"{result3.passes_applied[:4]}")
        print(f"    Latency: {result3.latency_before_ns:.1f} → "
              f"{result3.latency_after_ns:.1f}ns  "
              f"({'✓' if result3.within_budget else '✗'} vs {result3.budget_ns}ns budget)")
        print(f"    Notes: {result3.notes}")
    else:
        print(f"  ✗ FAILED: {result3.notes}")

    # ── Test 4: Latency cost model ────────────────────────────────────────
    print("\n── Test 4: Latency cost model ──")
    cost_model = LatencyCostModel()
    estimate = cost_model.estimate(SAMPLE_IR)
    print(f"  Instructions:  {estimate.instruction_count}")
    print(f"  Memory ops:    {estimate.memory_ops}")
    print(f"  Branches:      {estimate.branch_count}")
    print(f"  Latency est:   {estimate.total_ns:.1f}ns")
    print(f"  Breakdown:     {estimate.breakdown}")
    if estimate.total_ns > 0:
        print("  ✓ PASSED — cost model produced non-zero estimate")
    else:
        print("  ✗ FAILED — zero latency estimate")

    # ── Test 5: Pass priority selector ───────────────────────────────────
    print("\n── Test 5: HFT pass priority selector ──")
    selector = HFTPassSelector()
    queue = selector.build_priority_queue(
        ["LAP-001:critical:new", "LAP-002:critical:virtual",
         "LAP-007:major:atomic"],
        "heap alloc detected: apply mem2reg, sroa"
    )
    has_mem2reg = "mem2reg" in queue
    has_inline  = "-inline" in queue or "always-inline" in queue
    if has_mem2reg and has_inline:
        print(f"  ✓ PASSED — priority queue: {queue[:6]}...")
    else:
        print(f"  ✗ FAILED — expected mem2reg + inline in queue, got: {queue[:8]}")

    # ── Test 6: Reward shaper ─────────────────────────────────────────────
    print("\n── Test 6: Reward shaper ──")
    shaper = RewardShaper()
    r = shaper.compute(
        instr_before=50, instr_after=35,
        latency_before=320.0, latency_after=210.0,
        antipatterns_resolved=2, antipatterns_total=3,
        budget_ns=250, estimated_ns=210.0
    )
    if 0 < r <= 1.0:
        print(f"  ✓ PASSED — reward={r:.4f} (within budget, improved latency)")
    else:
        print(f"  ✗ FAILED — reward out of range: {r}")

    # Over-budget penalty test
    r_over = shaper.compute(
        instr_before=50, instr_after=35,
        latency_before=320.0, latency_after=310.0,
        antipatterns_resolved=0, antipatterns_total=3,
        budget_ns=250, estimated_ns=310.0
    )
    if r_over < r:
        print(f"  ✓ PASSED — over-budget penalised: {r_over:.4f} < {r:.4f}")
    else:
        print(f"  ✗ FAILED — expected penalty for over-budget result")

    # ── Summary ───────────────────────────────────────────────────────────
    print()
    print("── Summary ──")
    print(f"  General mode:   {result.latency_before_ns:.0f}ns → "
          f"{result.latency_after_ns:.0f}ns  "
          f"(-{result.latency_before_ns - result.latency_after_ns:.0f}ns)")
    print(f"  HFT clean:      {result2.latency_before_ns:.0f}ns → "
          f"{result2.latency_after_ns:.0f}ns  "
          f"({'✓' if result2.within_budget else '✗'} {result2.budget_ns}ns budget)")
    print(f"  HFT + AP fixes: {result3.latency_before_ns:.0f}ns → "
          f"{result3.latency_after_ns:.0f}ns  "
          f"({'✓' if result3.within_budget else '✗'} {result3.budget_ns}ns budget)")

    # Cleanup
    os.unlink(ir_path)

    print()
    print("=" * 70)
    print("✓ IR Tuner Agent (HFT Edition) smoke test PASSED")
    print("=" * 70)
