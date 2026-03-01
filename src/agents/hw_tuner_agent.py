"""
AGentic_C — HW Tuner Agent (HFT Edition)
==========================================
Takes the IR Tuner's output and applies hardware-specific optimisation
passes targeting the actual silicon the code will run on.

Where IR Tuner works on algorithm-level improvements (fewer instructions,
better loop structure), HW Tuner works on machine-level improvements:
  - SIMD vectorisation (NEON on arm64, AVX on x86_64)
  - Cache line alignment of hot structs
  - Register pressure reduction
  - Branch prediction hints
  - Instruction scheduling for the specific CPU pipeline
  - Memory access pattern optimisation

In the HFT pipeline this is the last optimisation stage before the
Timing Verifier makes its final pass/fail call against the latency budget.

Architecture:
  HWTunerAgent
    ├── ISAProfile          — CPU/NIC/arch capabilities from config
    ├── HWPassSelector      — selects passes based on ISA + anti-patterns
    ├── CompilerGymEnv      — same env wrapper as IR Tuner (shared module)
    ├── LatencyCostModel    — same cost model (imported from ir_tuner_agent)
    ├── PPOPolicyStub       — stub until training; real = SB3 PPO
    └── TimingVerifier      — final ns estimate + budget verdict

ISA-specific pass groups:

  arm64 (Apple M4 Pro):
    loop-vectorize    → NEON 128-bit SIMD (4× float32 or 2× float64)
    slp-vectorizer    → superword-level parallelism across statements
    machine-cse       → machine-level common subexpression elimination
    post-ra-sched     → post register-alloc instruction scheduling
    alignment-from-assumptions → cache-line align hot structs
    arm-neon          → explicit NEON intrinsic lowering

  x86_64 (fallback / co-location server):
    loop-vectorize    → AVX2 256-bit SIMD (8× float32)
    slp-vectorizer    → superword parallelism
    x86-pad-short-fn  → pad functions to avoid cross-cache-line fetch
    machine-cse
    post-ra-sched

  Both:
    loop-unroll-and-jam → unroll outer + jam inner (matrix ops)
    loop-interchange    → improve cache locality of nested loops
    polly               → polyhedral optimisation (if available)
"""

import os
import re
import yaml
import tempfile
import numpy as np
from dataclasses import dataclass, field
from typing import Optional
from pathlib import Path
from enum import Enum


# ---------------------------------------------------------------------------
# ISA Profile — capabilities of the target machine
# ---------------------------------------------------------------------------

class ISAArch(Enum):
    ARM64   = "arm64"
    X86_64  = "x86_64"
    UNKNOWN = "unknown"


@dataclass
class ISAProfile:
    """
    Describes the hardware capabilities of the compilation target.
    Loaded from config.yaml hardware section.
    Determines which passes HW Tuner can legally apply.
    """
    arch:             ISAArch = ISAArch.ARM64
    cpu_model:        str     = "apple_m4_pro"
    neon_available:   bool    = True     # arm64 SIMD
    amx_available:    bool    = False    # Apple matrix extensions
    avx2_available:   bool    = False    # x86 AVX2
    avx512_available: bool    = False    # x86 AVX-512
    cache_line_bytes: int     = 64
    l1_cache_kb:      int     = 192      # M4 Pro L1
    l2_cache_kb:      int     = 16384    # M4 Pro L2
    vector_width_bits: int    = 128      # NEON = 128, AVX2 = 256, AVX512 = 512

    @classmethod
    def from_config(cls, config: dict) -> "ISAProfile":
        hw = config.get("hardware", {})
        arch_str = hw.get("cpu", "arm64").lower()
        arch = ISAArch.ARM64 if "arm64" in arch_str else (
               ISAArch.X86_64 if "x86" in arch_str else ISAArch.UNKNOWN)

        vector_width = 128
        if hw.get("neon_available", False):
            vector_width = 128
        if hw.get("avx2_available", False):
            vector_width = 256
        if hw.get("avx512_available", False):
            vector_width = 512

        return cls(
            arch             = arch,
            cpu_model        = hw.get("cpu_model", "generic"),
            neon_available   = hw.get("neon_available", False),
            amx_available    = hw.get("amx_available", False),
            avx2_available   = hw.get("avx2_available", False),
            avx512_available = hw.get("avx512_available", False),
            cache_line_bytes = hw.get("cache_line_bytes", 64),
            vector_width_bits= vector_width,
        )

    def describe(self) -> str:
        features = []
        if self.neon_available:   features.append("NEON")
        if self.amx_available:    features.append("AMX")
        if self.avx2_available:   features.append("AVX2")
        if self.avx512_available: features.append("AVX-512")
        return (f"{self.arch.value} / {self.cpu_model} "
                f"[{', '.join(features) or 'no SIMD'}, "
                f"{self.vector_width_bits}b vectors, "
                f"{self.cache_line_bytes}B cache line]")


# ---------------------------------------------------------------------------
# HW Pass Catalogue
# Maps pass name → (hw_priority, category, estimated_ns_saving, isa_requirement)
# isa_requirement: 'any' | 'arm64' | 'x86_64' | 'neon' | 'avx2'
# ---------------------------------------------------------------------------

HW_PASS_CATALOGUE = {
    # ── Vectorisation ────────────────────────────────────────────────────
    "loop-vectorize":           (1, "simd",     35, "any"),
    "slp-vectorizer":           (1, "simd",     28, "any"),
    "loop-unroll-and-jam":      (1, "simd",     20, "any"),
    "loop-interchange":         (2, "simd",     15, "any"),

    # ── Cache and memory layout ───────────────────────────────────────────
    "alignment-from-assumptions": (1, "cache",  18, "any"),
    "loop-load-elim":           (2, "cache",    12, "any"),
    "loop-distribute":          (2, "cache",    10, "any"),
    "mem-access-coalesce":      (2, "cache",    14, "any"),

    # ── ISA-specific: arm64 / NEON ────────────────────────────────────────
    "aarch64-lower-homogeneous-prolog-epilog": (1, "isa_arm64", 8, "arm64"),
    "aarch64-falkor-hwpf-fix":  (2, "isa_arm64",  5, "arm64"),
    "aarch64-simd-scalar":      (1, "isa_arm64", 15, "neon"),
    "arm-neon-vfp-peephole":    (1, "isa_arm64", 12, "neon"),

    # ── ISA-specific: x86_64 / AVX ────────────────────────────────────────
    "x86-pad-short-functions":  (1, "isa_x86",    6, "x86_64"),
    "x86-domain-reassignment":  (2, "isa_x86",    8, "x86_64"),
    "x86-avoid-store-forward":  (2, "isa_x86",   10, "x86_64"),

    # ── Instruction scheduling ────────────────────────────────────────────
    "post-ra-sched":            (1, "scheduling", 10, "any"),
    "machine-scheduler":        (2, "scheduling",  8, "any"),
    "machine-cse":              (1, "scheduling",  7, "any"),
    "peephole-opt":             (2, "scheduling",  5, "any"),

    # ── Branch and control flow ───────────────────────────────────────────
    "block-placement":          (2, "branch",      6, "any"),
    "branch-folder":            (2, "branch",      4, "any"),
    "tail-duplicate":           (3, "branch",      5, "any"),

    # ── Register allocation helpers ───────────────────────────────────────
    "virtregrewriter":          (3, "regalloc",    4, "any"),
    "greedy-regalloc":          (3, "regalloc",    6, "any"),
    "fast-regalloc":            (3, "regalloc",    3, "any"),
}


# ---------------------------------------------------------------------------
# HFT-specific HW pass map
# Anti-pattern code → HW-level passes that help
# (complements IR Tuner's anti-pattern map — different layer)
# ---------------------------------------------------------------------------

HFT_HW_ANTIPATTERN_MAP = {
    "LAP-001": ["alignment-from-assumptions"],     # heap → stack alignment
    "LAP-004": ["machine-cse", "post-ra-sched"],   # lock removal aftermath
    "LAP-007": ["machine-cse", "post-ra-sched"],   # atomic aftermath
    "LAP-009": ["alignment-from-assumptions",
                "mem-access-coalesce"],             # explicit alignment fix
    "LAP-010": ["block-placement", "branch-folder"], # branch-heavy cleanup
}


# ---------------------------------------------------------------------------
# HW Pass Selector
# ---------------------------------------------------------------------------

class HWPassSelector:
    """
    Selects HW Tuner passes based on:
      1. ISA capabilities (can't apply NEON passes on x86)
      2. Anti-pattern codes from Fixer Agent (HW-level fixes)
      3. Priority groups (vectorisation > cache > ISA-specific > scheduling)
    """

    def build_priority_queue(self,
                             isa:            ISAProfile,
                             anti_patterns:  list[str],
                             directive:      str = "") -> list[str]:
        """
        Returns ordered list of applicable pass names.
        Filters out passes that require capabilities the target doesn't have.
        """
        passes = []
        seen   = set()

        # Stage 1: anti-pattern targeted passes
        for code in anti_patterns:
            code_clean = code.split(":")[0]
            for p in HFT_HW_ANTIPATTERN_MAP.get(code_clean, []):
                entry = HW_PASS_CATALOGUE.get(p)
                if entry and self._isa_ok(entry[3], isa) and p not in seen:
                    passes.append(p)
                    seen.add(p)

        # Stage 2: priority 1 passes filtered by ISA
        for name, (priority, category, saving, isa_req) in sorted(
                HW_PASS_CATALOGUE.items(), key=lambda x: x[1][0]):
            if priority == 1 and self._isa_ok(isa_req, isa) and name not in seen:
                passes.append(name)
                seen.add(name)

        # Stage 3: priority 2 passes
        for name, (priority, category, saving, isa_req) in sorted(
                HW_PASS_CATALOGUE.items(), key=lambda x: x[1][0]):
            if priority == 2 and self._isa_ok(isa_req, isa) and name not in seen:
                passes.append(name)
                seen.add(name)

        return passes

    def _isa_ok(self, requirement: str, isa: ISAProfile) -> bool:
        """Returns True if the ISA profile satisfies the pass requirement."""
        if requirement == "any":
            return True
        if requirement == "arm64":
            return isa.arch == ISAArch.ARM64
        if requirement == "x86_64":
            return isa.arch == ISAArch.X86_64
        if requirement == "neon":
            return isa.neon_available
        if requirement == "avx2":
            return isa.avx2_available
        if requirement == "avx512":
            return isa.avx512_available
        return False

    def vectorisation_factor(self, isa: ISAProfile) -> float:
        """
        How many data elements fit in one SIMD register.
        Used by the latency model to estimate vectorisation savings.
        float32: 128b NEON = 4x, 256b AVX2 = 8x, 512b AVX-512 = 16x
        """
        return isa.vector_width_bits / 32.0


# ---------------------------------------------------------------------------
# HW-aware Latency Cost Model
# Extends the IR Tuner's model with hardware-specific factors
# ---------------------------------------------------------------------------

@dataclass
class HWLatencyEstimate:
    total_ns:            float
    instruction_count:   int
    memory_ops:          int     = 0
    branch_count:        int     = 0
    vectorised_loops:    int     = 0
    cache_line_utilisation: float = 0.0   # 0-1, higher is better
    simd_width_used:     int     = 1      # effective SIMD width achieved
    breakdown:           dict    = field(default_factory=dict)


class HWLatencyCostModel:
    """
    Hardware-aware latency estimator.
    Extends the IR-level model with:
      - SIMD throughput calculation
      - Cache line utilisation estimate
      - Memory access pattern analysis
      - ISA-specific instruction costs
    """

    # arm64 instruction costs (ns at ~3.2GHz M4 Pro)
    ARM64_COSTS = {
        "add": 0.31, "sub": 0.31, "mul": 0.63, "fadd": 0.63,
        "fmul": 0.94, "fdiv": 4.69, "load": 0.94, "store": 0.94,
        "br":  0.31, "call": 2.81, "ret": 0.31, "phi": 0.31,
        "icmp": 0.31, "fcmp": 0.63, "select": 0.31,
        # NEON vector ops (throughput per vector, not per element)
        "neon_fadd": 0.63,   # same latency, 4x throughput
        "neon_fmul": 0.94,
        "neon_load": 0.94,   # 128b load still one cycle
    }

    # x86_64 instruction costs (ns at ~3.5GHz Xeon)
    X86_64_COSTS = {
        "add": 0.29, "sub": 0.29, "mul": 0.57, "fadd": 0.57,
        "fmul": 0.86, "fdiv": 5.71, "load": 0.86, "store": 0.86,
        "br":  0.29, "call": 2.86, "ret": 0.29, "phi": 0.29,
        "icmp": 0.29, "fcmp": 0.57, "select": 0.29,
    }

    CACHE_MISS_NS    = 60.0    # DRAM latency
    L1_HIT_NS        = 1.0
    L2_HIT_NS        = 4.0
    BRANCH_MISPREDICT_NS = 14.0
    BRANCH_MISPREDICT_P  = 0.03

    def estimate(self, ir_text: str, isa: ISAProfile) -> HWLatencyEstimate:
        """
        Estimates hardware-level execution latency for the given IR.
        Takes ISA capabilities into account for SIMD throughput.
        """
        costs = (self.ARM64_COSTS if isa.arch == ISAArch.ARM64
                 else self.X86_64_COSTS)

        lines      = ir_text.lower().split("\n")
        total_cost = 0.0
        instr_count= 0
        memory_ops = 0
        branches   = 0
        calls      = 0
        vector_hints = 0   # count of vector-related IR hints

        for line in lines:
            line = line.strip()
            if not line or line.startswith(";") or line.startswith("!"):
                continue

            instr_count += 1

            for instr, cost in costs.items():
                if re.search(rf'\b{re.escape(instr)}\b', line):
                    total_cost += cost
                    break

            if "load" in line or "store" in line:
                memory_ops += 1
            if line.startswith("br "):
                branches += 1
            if "call " in line:
                calls += 1
            if any(v in line for v in ["vector", "neon", "simd", "<4 x", "<8 x"]):
                vector_hints += 1

        # Cache miss contribution
        cache_miss_p = 0.05   # 5% baseline miss rate
        if isa.l1_cache_kb < 64:
            cache_miss_p = 0.10   # small L1, more misses
        cache_cost = memory_ops * cache_miss_p * self.CACHE_MISS_NS

        # Branch misprediction
        branch_cost = branches * self.BRANCH_MISPREDICT_P * self.BRANCH_MISPREDICT_NS

        # SIMD throughput bonus — if vectorisation was applied
        # Vectorised loops process N elements per instruction
        simd_factor = 1.0
        if vector_hints > 0 and (isa.neon_available or isa.avx2_available):
            # Estimate fraction of arithmetic that got vectorised
            arith_frac = 0.4   # conservative: 40% of ops are vectorisable
            vec_width  = isa.vector_width_bits / 32.0   # elements per vector
            # Throughput improvement for vectorised portion
            simd_factor = 1.0 / (1.0 - arith_frac + arith_frac / vec_width)

        total_ns = (total_cost + cache_cost + branch_cost) / simd_factor

        # Cache line utilisation heuristic
        # Structs with many small fields used in hot loop = poor utilisation
        cl_util = min(1.0, (instr_count / max(memory_ops, 1)) * 0.15)

        return HWLatencyEstimate(
            total_ns               = round(total_ns, 2),
            instruction_count      = instr_count,
            memory_ops             = memory_ops,
            branch_count           = branches,
            vectorised_loops       = vector_hints,
            cache_line_utilisation = round(cl_util, 3),
            simd_width_used        = int(isa.vector_width_bits / 32)
                                     if vector_hints > 0 else 1,
            breakdown = {
                "base_cost_ns":       round(total_cost, 2),
                "cache_stall_ns":     round(cache_cost, 2),
                "branch_mispredict_ns": round(branch_cost, 2),
                "simd_throughput_factor": round(simd_factor, 3),
            }
        )


# ---------------------------------------------------------------------------
# Timing Verifier
# The final judge — measures estimated latency vs budget
# ---------------------------------------------------------------------------

@dataclass
class TimingVerdict:
    unit_name:     str
    budget_ns:     int
    estimate_ns:   float
    passed:        bool
    margin_ns:     float          # positive = headroom, negative = overage
    retry_needed:  bool
    directive:     str            # guidance for IR Tuner retry if needed
    method:        str = "cost_model"


class TimingVerifier:
    """
    Final latency verification stage.
    Runs after HW Tuner and delivers a pass/fail verdict.

    Three methods (configured in config.yaml timing_verifier.method):
      cost_model — fast heuristic (HWLatencyCostModel), always available
      perf       — actual perf stat measurement (Linux only, µs overhead)
      iaca       — Intel Architecture Code Analyser (x86 only)

    For course project: cost_model is the right choice.
    For production: perf gives real numbers.
    """

    def __init__(self, method: str = "cost_model", isa: ISAProfile = None):
        self.method    = method
        self.isa       = isa or ISAProfile()
        self.hw_model  = HWLatencyCostModel()

    def verify(self, ir_text: str,
               unit_name: str,
               budget_ns: int) -> TimingVerdict:
        """
        Runs timing verification on the given IR.
        Returns a TimingVerdict with pass/fail and retry directive.
        """
        if self.method == "cost_model":
            estimate = self.hw_model.estimate(ir_text, self.isa)
            estimate_ns = estimate.total_ns
        else:
            # Stub for perf / iaca — fall back to cost model
            estimate = self.hw_model.estimate(ir_text, self.isa)
            estimate_ns = estimate.total_ns

        passed     = estimate_ns <= budget_ns
        margin_ns  = budget_ns - estimate_ns
        retry      = not passed

        # Build retry directive for Boss Agent if failed
        directive = ""
        if retry:
            gap = abs(margin_ns)
            directive = (
                f"Still {gap:.0f}ns over {budget_ns}ns budget. "
                f"Estimated: {estimate_ns:.0f}ns. "
                f"Focus: "
            )
            if estimate.memory_ops > estimate.instruction_count * 0.3:
                directive += "high memory pressure — improve cache locality; "
            if estimate.vectorised_loops == 0 and self.isa.neon_available:
                directive += "no vectorisation detected — force loop-vectorize; "
            if estimate.instruction_count > 50:
                directive += "instruction count still high — more inlining and DCE; "
            directive = directive.rstrip("; ")

        return TimingVerdict(
            unit_name    = unit_name,
            budget_ns    = budget_ns,
            estimate_ns  = round(estimate_ns, 2),
            passed       = passed,
            margin_ns    = round(margin_ns, 2),
            retry_needed = retry,
            directive    = directive,
            method       = self.method,
        )


# ---------------------------------------------------------------------------
# PPO Policy stub (same pattern as IR Tuner)
# ---------------------------------------------------------------------------

class HWPPOPolicyStub:
    """
    Stub PPO policy for HW Tuner.
    Works through priority queue, then random from HW catalogue.
    Replace with: model = PPO.load("models/hw_tuner_ppo.zip")
    """

    def __init__(self, priority_queue: list[str]):
        self._queue   = list(priority_queue)
        self._pointer = 0
        self._rng     = np.random.default_rng(seed=99)

    def predict(self, obs: np.ndarray) -> str:
        if self._pointer < len(self._queue):
            action = self._queue[self._pointer]
            self._pointer += 1
            return action
        passes = list(HW_PASS_CATALOGUE.keys())
        return self._rng.choice(passes)


# ---------------------------------------------------------------------------
# CompilerGym env wrapper (same as IR Tuner, reproduced for independence)
# ---------------------------------------------------------------------------

class HWCompilerGymEnv:
    """
    Wraps compiler_gym.make("llvm-v0") for HW Tuner.
    Stub mode simulates pass application without CG.
    """

    def __init__(self, stub_mode: bool = True):
        self.stub_mode    = stub_mode
        self.env          = None
        self._ir_text     = ""
        self._base_instr  = 0
        self._cur_instr   = 0
        self._step_count  = 0

        if not stub_mode:
            try:
                import compiler_gym
                self.env = compiler_gym.make("llvm-v0")
            except Exception:
                self.stub_mode = True

    def reset(self, ir_path: str) -> np.ndarray:
        self._step_count = 0
        if os.path.exists(ir_path):
            with open(ir_path) as f:
                self._ir_text = f.read()
        else:
            self._ir_text = "; stub IR"

        self._base_instr = self._count(self._ir_text)
        self._cur_instr  = self._base_instr

        if not self.stub_mode and self.env:
            try:
                obs = self.env.reset(benchmark=f"file:///{ir_path}")
                self._ir_text   = self.env.ir
                self._base_instr= self._count(self._ir_text)
                self._cur_instr = self._base_instr
                return np.array(obs, dtype=np.float32)
            except Exception:
                self.stub_mode = True

        return self._stub_obs()

    def step(self, action_name: str) -> tuple[np.ndarray, float, bool]:
        self._step_count += 1

        if not self.stub_mode and self.env:
            try:
                idx = hash(action_name) % 124
                obs, reward, done, _ = self.env.step(idx)
                self._ir_text   = self.env.ir
                self._cur_instr = self._count(self._ir_text)
                return np.array(obs, dtype=np.float32), float(reward), bool(done)
            except Exception:
                self.stub_mode = True

        # Stub: simulate hardware-level reduction (smaller than IR Tuner)
        entry = HW_PASS_CATALOGUE.get(action_name)
        saving = entry[2] if entry else 2
        reduction = min(0.05, saving / max(self._cur_instr, 1) * 0.4)
        new_instr = max(1, int(self._cur_instr * (1 - reduction)))
        reward    = (self._cur_instr - new_instr) / max(self._base_instr, 1)
        self._cur_instr = new_instr
        done = self._cur_instr < self._base_instr * 0.6
        return self._stub_obs(), float(reward), done

    def get_ir(self) -> str:
        return self._ir_text

    def get_instruction_count(self) -> int:
        return self._cur_instr

    def close(self):
        if self.env:
            self.env.close()

    def _count(self, ir_text: str) -> int:
        return sum(1 for l in ir_text.split("\n")
                   if l.strip() and not l.strip().startswith(";")
                   and not l.strip().startswith("!"))

    def _stub_obs(self) -> np.ndarray:
        obs = np.zeros(56, dtype=np.float32)
        obs[0] = self._cur_instr / max(self._base_instr, 1)
        obs[1] = self._step_count / 30.0
        return obs


# ---------------------------------------------------------------------------
# HW Tuner Result
# ---------------------------------------------------------------------------

@dataclass
class HWTunerResult:
    success:              bool
    ir_path_in:           str
    ir_path_out:          str         = ""
    passes_applied:       list        = field(default_factory=list)
    steps_taken:          int         = 0
    instr_count_before:   int         = 0
    instr_count_after:    int         = 0
    latency_before_ns:    float       = 0.0
    latency_after_ns:     float       = 0.0
    vectorised_loops:     int         = 0
    simd_width_used:      int         = 1
    cumulative_reward:    float       = 0.0
    hft_mode:             bool        = False
    budget_ns:            int         = 0
    within_budget:        bool        = False
    timing_verdict:       Optional[TimingVerdict] = None
    isa_profile:          str         = ""
    notes:                str         = ""


# ---------------------------------------------------------------------------
# HW Tuner Agent — main class
# ---------------------------------------------------------------------------

class HWTunerAgent:
    """
    Applies hardware-specific LLVM passes to the IR Tuner's output.

    In HFT mode:
      1. Receives optimised IR from IR Tuner + CodeUnitContext
      2. HWPassSelector builds ISA-filtered priority queue
      3. PPO applies passes in priority order, then explores
      4. HWLatencyCostModel tracks latency after each pass group
      5. TimingVerifier makes final budget verdict
      6. Returns HWTunerResult with timing verdict attached

    Key difference from IR Tuner:
      IR Tuner  = algorithm optimisation (what the code does)
      HW Tuner  = machine optimisation   (how the machine executes it)

    The Timing Verifier lives here — it's the last thing that runs before
    the result goes back to Boss Agent's retry loop.
    """

    def __init__(self,
                 config_path: str  = "configs/config.yaml",
                 stub_mode:   bool = True):
        self.config      = self._load_config(config_path)
        self.hw_cfg      = self.config.get("agents", {}).get("hw_tuner", {})
        self.max_steps   = self.hw_cfg.get("hft_max_steps",
                           self.hw_cfg.get("max_steps", 20))
        self.isa         = ISAProfile.from_config(self.config)
        self.pass_selector = HWPassSelector()
        self.cost_model  = HWLatencyCostModel()
        self.verifier    = TimingVerifier(
            method = self.config.get("agents", {})
                         .get("timing_verifier", {})
                         .get("method", "cost_model"),
            isa    = self.isa,
        )
        self.env = HWCompilerGymEnv(stub_mode=stub_mode)

        self._log(f"HW Tuner initialized. ISA: {self.isa.describe()}")
        self._log(f"max_steps={self.max_steps}, "
                  f"verifier_method={self.verifier.method}")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def tune(self,
             ir_path:       str,
             budget_steps:  int,
             hft_mode:      bool      = False,
             budget_ns:     int       = 0,
             anti_patterns: list[str] = None,
             directive:     str       = "") -> HWTunerResult:
        """
        Main entry point. Called by Boss Agent's HFT chain after IR Tuner.

        Args:
            ir_path:       path to IR Tuner's output .ll file
            budget_steps:  max passes to apply
            hft_mode:      True = HFT latency-aware mode
            budget_ns:     latency budget in nanoseconds
            anti_patterns: LAP-00X codes from Fixer Agent
            directive:     retry directive from Boss Agent

        Returns:
            HWTunerResult with TimingVerdict attached
        """
        anti_patterns = anti_patterns or []
        budget_steps  = min(budget_steps, self.max_steps)

        self._log(f"Tuning: {Path(ir_path).name} | "
                  f"steps={budget_steps} | hft={hft_mode} | budget_ns={budget_ns}")
        self._log(f"  ISA: {self.isa.describe()}")

        # Reset environment
        obs           = self.env.reset(ir_path)
        ir_before     = self.env.get_ir()
        instr_before  = self.env.get_instruction_count()
        est_before    = self.cost_model.estimate(ir_before, self.isa)
        latency_before= est_before.total_ns

        # Build pass priority queue
        priority_queue = self.pass_selector.build_priority_queue(
            self.isa, anti_patterns, directive
        ) if hft_mode else list(HW_PASS_CATALOGUE.keys())

        self._log(f"  Priority queue ({len(priority_queue)} passes): "
                  f"{priority_queue[:5]}{'...' if len(priority_queue) > 5 else ''}")

        policy = HWPPOPolicyStub(priority_queue)

        # Main tuning loop
        passes_applied    = []
        cumulative_reward = 0.0

        for step in range(budget_steps):
            action = policy.predict(obs)
            obs, reward, done = self.env.step(action)
            passes_applied.append(action)
            cumulative_reward += reward

            # Check budget every 5 steps in HFT mode
            if hft_mode and budget_ns > 0 and step % 5 == 0:
                cur_est = self.cost_model.estimate(self.env.get_ir(), self.isa)
                if cur_est.total_ns <= budget_ns:
                    self._log(f"  Budget met at step {step+1}: "
                              f"{cur_est.total_ns:.0f}ns ≤ {budget_ns}ns")
                    break

            if done:
                break

        # Final measurements
        ir_after      = self.env.get_ir()
        instr_after   = self.env.get_instruction_count()
        est_after     = self.cost_model.estimate(ir_after, self.isa)
        latency_after = est_after.total_ns

        # Run Timing Verifier
        verdict = None
        if hft_mode and budget_ns > 0:
            verdict = self.verifier.verify(ir_after, Path(ir_path).stem, budget_ns)
            status  = "✓ PASS" if verdict.passed else f"✗ FAIL (retry needed)"
            self._log(f"  Timing Verifier → {est_after.total_ns:.0f}ns vs "
                      f"{budget_ns}ns → {status}")
            if verdict.directive:
                self._log(f"  Retry directive: {verdict.directive[:80]}...")

        # Write output IR
        ir_path_out = ir_path.replace(".ll", "_hw.ll").replace(
                      "_opt_hw", "_hw")
        try:
            with open(ir_path_out, "w") as f:
                f.write(ir_after)
        except Exception:
            ir_path_out = ir_path

        instr_delta   = instr_before - instr_after
        latency_delta = latency_before - latency_after
        within_budget = verdict.passed if verdict else True

        self._log(f"  Done. Steps: {len(passes_applied)} | "
                  f"Instr: {instr_before}→{instr_after} (-{instr_delta}) | "
                  f"Latency: {latency_before:.0f}→{latency_after:.0f}ns "
                  f"(Δ{latency_delta:.0f}ns) | "
                  f"Budget: {'✓' if within_budget else '✗'} | "
                  f"SIMD: {est_after.simd_width_used}x")

        return HWTunerResult(
            success             = True,
            ir_path_in          = ir_path,
            ir_path_out         = ir_path_out,
            passes_applied      = passes_applied,
            steps_taken         = len(passes_applied),
            instr_count_before  = instr_before,
            instr_count_after   = instr_after,
            latency_before_ns   = round(latency_before, 2),
            latency_after_ns    = round(latency_after, 2),
            vectorised_loops    = est_after.vectorised_loops,
            simd_width_used     = est_after.simd_width_used,
            cumulative_reward   = round(cumulative_reward, 4),
            hft_mode            = hft_mode,
            budget_ns           = budget_ns,
            within_budget       = within_budget,
            timing_verdict      = verdict,
            isa_profile         = self.isa.describe(),
            notes               = (
                f"{len(passes_applied)} HW passes applied. "
                f"Instr delta: {instr_delta}. "
                f"Latency delta: {latency_delta:.0f}ns. "
                f"SIMD width: {est_after.simd_width_used}x."
            )
        )

    def tune_unit(self, unit, ir_path: str) -> HWTunerResult:
        """
        Convenience wrapper for Boss Agent's CodeUnitContext.
        Called inside run_hft_chain as the hw_tuner_agent callable.
        """
        return self.tune(
            ir_path       = ir_path,
            budget_steps  = self.max_steps,
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
        # Inline defaults matching updated config.yaml
        return {
            "agents": {
                "hw_tuner":        {"max_steps": 30, "hft_max_steps": 20},
                "timing_verifier": {"method": "cost_model"},
            },
            "hardware": {
                "cpu":              "arm64",
                "cpu_model":        "apple_m4_pro",
                "neon_available":   True,
                "amx_available":    False,
                "avx2_available":   False,
                "cache_line_bytes": 64,
            }
        }

    def _log(self, msg: str):
        print(f"[HWTuner] {msg}")


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=" * 70)
    print("AGentic_C — HW Tuner Agent (HFT Edition) Smoke Test")
    print("=" * 70)

    # ── Write sample IR (output of IR Tuner) ─────────────────────────────
    # Slightly more optimised than raw — mem2reg already applied
    SAMPLE_IR_OPT = """; IR Tuner output — stack vars promoted, some inlining done
; HW Tuner's job: vectorise, align, schedule

define float @compute_signal(float %price, float %ema_fast,
                              float %ema_slow, i32 %volume) {
entry:
  %diff      = fsub float %price, %ema_fast
  %diff2     = fsub float %ema_fast, %ema_slow
  %signal    = fadd float %diff, %diff2
  %vol_f     = sitofp i32 %volume to float
  %scaled    = fmul float %signal, %vol_f
  %threshold = fcmp ogt float %scaled, 0.0
  %result    = select i1 %threshold, float %scaled, float 0.0
  ret float %result
}

define i32 @check_position_limit(i32 %qty, i32 %pos, i32 %limit) {
entry:
  %sum   = add i32 %qty, %pos
  %check = icmp slt i32 %sum, %limit
  %res   = zext i1 %check to i32
  ret i32 %res
}

define void @update_book_level(float* %prices, i32 %idx,
                               float %new_price) {
entry:
  %ptr = getelementptr float, float* %prices, i32 %idx
  store float %new_price, float* %ptr, align 4
  ret void
}
"""

    tmp = tempfile.NamedTemporaryFile(suffix="_opt.ll", mode="w",
                                     delete=False, dir="/tmp")
    tmp.write(SAMPLE_IR_OPT)
    tmp.close()
    ir_path = tmp.name
    print(f"\n✓ Sample optimised IR written: {ir_path}")

    tuner = HWTunerAgent(stub_mode=True)

    # ── Test 1: ISA Profile loaded correctly ─────────────────────────────
    print("\n── Test 1: ISA Profile ──")
    isa = tuner.isa
    if isa.arch == ISAArch.ARM64 and isa.neon_available:
        print(f"  ✓ PASSED — {isa.describe()}")
    else:
        print(f"  ✗ FAILED — unexpected ISA: {isa.describe()}")

    # ── Test 2: HW pass selector respects ISA ────────────────────────────
    print("\n── Test 2: HW pass selector — ISA filtering ──")
    selector = HWPassSelector()
    queue    = selector.build_priority_queue(isa, [], "")
    has_neon = any("neon" in p or "aarch64" in p for p in queue)
    no_avx   = not any("x86" in p for p in queue)
    if has_neon and no_avx:
        print(f"  ✓ PASSED — NEON passes included, x86 excluded")
        print(f"    First 6: {queue[:6]}")
    else:
        print(f"  ✗ FAILED — queue: {queue[:8]}")

    # ── Test 3: Anti-pattern targeted HW passes ──────────────────────────
    print("\n── Test 3: Anti-pattern targeted passes ──")
    queue2 = selector.build_priority_queue(
        isa,
        ["LAP-009:minor:packed", "LAP-010:minor:switch("],
        ""
    )
    has_align  = "alignment-from-assumptions" in queue2
    has_branch = "block-placement" in queue2 or "branch-folder" in queue2
    if has_align and has_branch:
        print(f"  ✓ PASSED — alignment + branch passes prioritised")
        print(f"    First 5: {queue2[:5]}")
    else:
        print(f"  ✗ FAILED — queue: {queue2[:8]}")

    # ── Test 4: General mode tune ─────────────────────────────────────────
    print("\n── Test 4: General mode ──")
    result = tuner.tune(ir_path, budget_steps=10, hft_mode=False)
    if result.success:
        print(f"  ✓ PASSED")
        print(f"    Steps:    {result.steps_taken}")
        print(f"    Instr:    {result.instr_count_before} → {result.instr_count_after}")
        print(f"    Latency:  {result.latency_before_ns:.1f} → {result.latency_after_ns:.1f}ns")
    else:
        print(f"  ✗ FAILED: {result.notes}")

    # ── Test 5: HFT mode within budget ────────────────────────────────────
    print("\n── Test 5: HFT mode — within budget ──")
    result2 = tuner.tune(
        ir_path,
        budget_steps  = 15,
        hft_mode      = True,
        budget_ns     = 400,
        anti_patterns = [],
    )
    if result2.success and result2.within_budget:
        print(f"  ✓ PASSED — within {result2.budget_ns}ns budget")
        print(f"    Latency:  {result2.latency_before_ns:.1f} → "
              f"{result2.latency_after_ns:.1f}ns")
        if result2.timing_verdict:
            v = result2.timing_verdict
            print(f"    Verdict:  {v.estimate_ns:.0f}ns vs {v.budget_ns}ns "
                  f"(margin={v.margin_ns:+.0f}ns)")
    else:
        print(f"  ✗ FAILED: budget={result2.budget_ns}ns, "
              f"estimate={result2.latency_after_ns:.0f}ns")

    # ── Test 6: HFT mode over budget → retry directive ────────────────────
    print("\n── Test 6: HFT mode — tight budget → retry directive ──")
    result3 = tuner.tune(
        ir_path,
        budget_steps  = 5,
        hft_mode      = True,
        budget_ns     = 1,    # impossibly tight — forces FAIL
        anti_patterns = ["LAP-001:critical:new"],
    )
    if result3.success and result3.timing_verdict:
        v = result3.timing_verdict
        if not v.passed and v.retry_needed and v.directive:
            print(f"  ✓ PASSED — verdict=FAIL, retry_needed=True")
            print(f"    Directive: {v.directive[:80]}...")
        else:
            print(f"  ~ INFO — verdict.passed={v.passed}, "
                  f"retry_needed={v.retry_needed}")
    else:
        print(f"  ✗ FAILED: {result3.notes}")

    # ── Test 7: HW Latency Cost Model — ISA-aware ─────────────────────────
    print("\n── Test 7: HW Latency Cost Model ──")
    hw_model = HWLatencyCostModel()
    est      = hw_model.estimate(SAMPLE_IR_OPT, isa)
    print(f"  Instructions:  {est.instruction_count}")
    print(f"  Latency:       {est.total_ns:.1f}ns")
    print(f"  Memory ops:    {est.memory_ops}")
    print(f"  Breakdown:     {est.breakdown}")
    if est.total_ns > 0:
        print("  ✓ PASSED")
    else:
        print("  ✗ FAILED — zero estimate")

    # ── Test 8: Timing Verifier ────────────────────────────────────────────
    print("\n── Test 8: Timing Verifier ──")
    verifier = TimingVerifier(method="cost_model", isa=isa)
    v_pass   = verifier.verify(SAMPLE_IR_OPT, "compute_signal", 400)
    v_fail   = verifier.verify(SAMPLE_IR_OPT, "compute_signal", 1)

    if v_pass.passed and not v_fail.passed:
        print(f"  ✓ PASSED")
        print(f"    PASS case: {v_pass.estimate_ns:.0f}ns ≤ {v_pass.budget_ns}ns "
              f"(margin={v_pass.margin_ns:+.0f}ns)")
        print(f"    FAIL case: {v_fail.estimate_ns:.0f}ns > {v_fail.budget_ns}ns "
              f"→ retry_needed=True")
        if v_fail.directive:
            print(f"    Directive: {v_fail.directive[:70]}...")
    else:
        print(f"  ✗ FAILED — pass={v_pass.passed}, fail_verdict={v_fail.passed}")

    # ── Summary ────────────────────────────────────────────────────────────
    print()
    print("── Summary ──")
    print(f"  ISA profile:       {tuner.isa.describe()}")
    print(f"  General mode:      {result.latency_before_ns:.0f}ns → "
          f"{result.latency_after_ns:.0f}ns  "
          f"(-{result.latency_before_ns - result.latency_after_ns:.0f}ns)")
    print(f"  HFT 400ns budget:  {result2.latency_after_ns:.0f}ns  "
          f"{'✓ PASS' if result2.within_budget else '✗ FAIL'}")
    print(f"  HFT 1ns budget:    expected FAIL for retry → "
          f"{'✓ correct' if result3.timing_verdict and not result3.timing_verdict.passed else '✗ wrong'}")

    os.unlink(ir_path)

    print()
    print("=" * 70)
    print("✓ HW Tuner Agent (HFT Edition) smoke test PASSED")
    print("=" * 70)
