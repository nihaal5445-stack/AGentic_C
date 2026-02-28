"""
AGentic_C — Boss Agent (HFT Edition)
=====================================
Orchestrates the full compilation pipeline with HFT-aware routing.

New in this version:
  - PathClassifier     : labels each code unit HOT or COLD
  - LatencyBudget      : per-unit ns budgets loaded from profile yaml
  - CodeUnitContext    : context packet passed through the agent chain
  - HFT agent chain   : Fixer → IR Tuner → HW Tuner → Timing Verifier
  - Budget retry loop : re-routes to IR Tuner if timing estimate fails
  - Cold path chain   : unchanged general pipeline for non-critical code
"""

import os
import re
import ast
import yaml
import numpy as np
from dataclasses import dataclass, field
from typing import Optional, Literal
from pathlib import Path
from enum import Enum


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class PathLabel(Enum):
    HOT      = "hot"       # on the critical event-driven path
    COLD     = "cold"      # startup, teardown, config, logging
    UNKNOWN  = "unknown"   # could not determine — treated conservatively as HOT


class ChainResult(Enum):
    PASS    = "pass"       # within budget
    FAIL    = "fail"       # over budget after max retries
    SKIPPED = "skipped"    # cold path — no budget enforcement


# ---------------------------------------------------------------------------
# HFT latency budget — loaded from profile yaml
# ---------------------------------------------------------------------------

@dataclass
class LatencyBudget:
    """
    Per-unit nanosecond budgets for hot path code.
    Loaded from configs/hft_profile.yaml.
    All values in nanoseconds.
    """
    tick_to_order_ns:        int = 2000   # total hot path budget
    market_data_parse_ns:    int = 200    # parse the wire message
    signal_eval_ns:          int = 400    # evaluate alpha/signal condition
    risk_check_ns:           int = 150    # inline risk gate
    order_serialise_ns:      int = 250    # build + send the order
    headroom_ns:             int = 200    # buffer for jitter / NIC variance

    # Derived — total accounted budget (should be <= tick_to_order_ns)
    @property
    def accounted_ns(self) -> int:
        return (self.market_data_parse_ns
                + self.signal_eval_ns
                + self.risk_check_ns
                + self.order_serialise_ns
                + self.headroom_ns)

    def budget_for(self, unit_tag: str) -> int:
        """
        Returns the sub-budget for a labelled code unit.
        unit_tag should match one of the known hot path segments.
        Falls back to a conservative fraction of total budget.
        """
        mapping = {
            "market_data_parse":  self.market_data_parse_ns,
            "signal_eval":        self.signal_eval_ns,
            "risk_check":         self.risk_check_ns,
            "order_serialise":    self.order_serialise_ns,
        }
        return mapping.get(unit_tag, self.tick_to_order_ns // 4)


# ---------------------------------------------------------------------------
# Hardware system profile — NIC, NUMA, exchange target
# ---------------------------------------------------------------------------

@dataclass
class HardwareProfile:
    """
    Complete system profile for HFT targeting.
    General compilers only know CPU arch.
    We also need NIC model, NUMA topology, and exchange protocol.
    """
    cpu:             str = "x86_64"              # e.g. 'Xeon_8375C', 'arm64'
    nic:             str = "generic"             # e.g. 'Solarflare_X2', 'Mellanox_CX6'
    bypass_mode:     str = "none"                # 'none' | 'dpdk' | 'openonload' | 'vma'
    numa_node:       int = 0
    cpu_isolated:    bool = False                # is the core pinned and isolated?
    exchange_proto:  str = "fix"                 # 'fix' | 'itch' | 'ouch' | 'sbe'
    arch_string:     str = "x86_64-linux-gnu"    # LLVM target triple


# ---------------------------------------------------------------------------
# Per-unit context packet — passed through the entire agent chain
# ---------------------------------------------------------------------------

@dataclass
class CodeUnitContext:
    """
    Everything the agent chain knows about a single code unit (function/block).
    Every agent reads this, does its work, and writes results back into it.
    The Boss Agent monitors it and decides retry vs. finalise.
    """
    unit_name:          str                          # function or block name
    source_snippet:     str                          # raw source text
    path_label:         PathLabel = PathLabel.UNKNOWN
    unit_tag:           str = "unknown"              # 'signal_eval', 'risk_check', etc.
    budget_ns:          int = 0                      # assigned by Boss after classification
    hardware_profile:   Optional[HardwareProfile] = None

    # Populated by Fixer Agent
    anti_patterns:      list = field(default_factory=list)
    fixer_notes:        str = ""

    # Populated by IR Tuner
    ir_optimised:       bool = False
    ir_tuner_notes:     str = ""

    # Populated by Timing Verifier
    timing_estimate_ns: Optional[int] = None
    passed_budget:      Optional[bool] = None        # None = not yet verified

    # Retry tracking
    retry_count:        int = 0
    max_retries:        int = 3
    chain_result:       ChainResult = ChainResult.SKIPPED


# ---------------------------------------------------------------------------
# Data structures (extended from original)
# ---------------------------------------------------------------------------

@dataclass
class CompilationContext:
    """Everything the Boss Agent knows about an incoming compilation job."""
    source_path:        str
    source_lang:        str                         # 'c', 'cpp', 'rust'
    target_arch:        str                         # 'arm64', 'x86_64'
    ir_embedding:       Optional[np.ndarray]        # 256-dim float vector
    optimization_budget: int = 45
    past_experiences:   list = field(default_factory=list)

    # HFT additions
    hft_mode:           bool = False                # enable HFT agent chain
    latency_budget:     Optional[LatencyBudget] = None
    hardware_profile:   Optional[HardwareProfile] = None
    code_units:         list = field(default_factory=list)  # List[CodeUnitContext]


@dataclass
class CompilationPlan:
    """Structured decision output from Boss Agent."""
    # Pre-IR Fixer
    run_pre_fixer:      bool = True
    pre_fixer_focus:    str  = "syntax"             # 'syntax' | 'security' | 'both' | 'latency'

    # Post-IR Fixer
    run_post_fixer:     bool = True
    post_fixer_focus:   str  = "security"

    # IR Tuner
    run_ir_tuner:       bool = True
    ir_tuner_budget:    int  = 30
    ir_tuner_directive: str  = ""                   # HFT: tighter re-route instructions

    # HW Tuner
    run_hw_tuner:       bool = True
    hw_tuner_budget:    int  = 15
    hw_target:          str  = "arm64-apple-macosx"

    # Timing Verifier (HFT only)
    run_timing_verifier: bool = False

    # Routing metadata
    confidence:         float = 0.0
    based_on_memory:    bool  = False
    retry_count:        int   = 0

    # HFT additions
    hft_chain_active:   bool  = False               # True when routing hot path units
    hot_units:          list  = field(default_factory=list)   # List[CodeUnitContext]
    cold_units:         list  = field(default_factory=list)   # List[CodeUnitContext]


# ---------------------------------------------------------------------------
# Config loader
# ---------------------------------------------------------------------------

def load_config(config_path: str = "configs/config.yaml") -> dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def load_hft_profile(profile_path: str = "configs/hft_profile.yaml") -> dict:
    """
    Loads the HFT latency budget and hardware profile from yaml.
    Returns empty dict if file not found — caller handles gracefully.
    """
    if not os.path.exists(profile_path):
        return {}
    with open(profile_path, "r") as f:
        return yaml.safe_load(f)


# ---------------------------------------------------------------------------
# IR Encoder (unchanged from original — lightweight Autophase-style)
# ---------------------------------------------------------------------------

class SimpleIREncoder:
    """
    Encodes LLVM IR text into a fixed-size float vector.
    Lightweight version — no deep learning overhead on the Boss Agent.
    Full GNN version lives in src/encoder/ir_encoder.py.
    """

    DIM = 256

    INSTRUCTION_TYPES = [
        "alloca", "load", "store", "add", "sub", "mul", "sdiv", "udiv",
        "fadd", "fsub", "fmul", "fdiv", "icmp", "fcmp", "br", "ret",
        "call", "phi", "select", "getelementptr", "bitcast", "zext",
        "sext", "trunc", "and", "or", "xor", "shl", "lshr", "ashr",
        "switch", "invoke", "unreachable", "extractvalue", "insertvalue",
    ]

    def encode(self, ir_text: str) -> np.ndarray:
        vec   = np.zeros(self.DIM, dtype=np.float32)
        lines = ir_text.lower().split("\n")
        total = max(len(lines), 1)

        for i, instr in enumerate(self.INSTRUCTION_TYPES):
            vec[i] = sum(1 for l in lines if instr in l) / total

        vec[35] = ir_text.count("define ")  / 100.0
        vec[36] = ir_text.count("declare ") / 100.0
        vec[37] = total                     / 1000.0
        vec[38] = ir_text.count("phi")      / max(ir_text.count("define "), 1)
        vec[39] = ir_text.count("call")     / total
        vec[40] = ir_text.count("br")       / total
        vec[41] = ir_text.count("loop")     / total
        vec[42] = ir_text.count("ptr")      / total
        vec[43] = ir_text.count("global")   / 100.0
        vec[44] = ir_text.count("metadata") / 100.0

        return vec


# ---------------------------------------------------------------------------
# Path Classifier — the new core of the HFT Boss Agent
# ---------------------------------------------------------------------------

class PathClassifier:
    """
    Classifies each code unit as HOT or COLD.

    Two modes, applied in order:
      1. Annotation-driven  : reads [[hft::hot]] / [[hft::cold]] attributes
      2. Inference-driven   : static call graph traversal from known hot roots

    Known hot roots — functions that are always hot entry points:
      - on_market_data, on_tick, on_quote, on_trade
      - on_order_update, on_execution_report
      - event_loop, poll_feed, process_message

    Known cold roots — functions that are always cold:
      - on_connect, on_disconnect, on_session_*
      - load_config, init_*, shutdown_*, cleanup_*
      - log_*, audit_*, persist_*
    """

    # Annotation patterns
    HOT_ANNOTATION  = re.compile(r'\[\[hft::hot\]\]')
    COLD_ANNOTATION = re.compile(r'\[\[hft::cold\]\]')

    # Known entry point roots
    HOT_ROOTS = {
        "on_market_data", "on_tick", "on_quote", "on_trade",
        "on_order_update", "on_execution_report",
        "event_loop", "poll_feed", "process_message",
        "handle_tick", "dispatch_order", "submit_order",
        "evaluate_signal", "check_risk", "send_order",
    }

    COLD_ROOTS = {
        "on_connect", "on_disconnect", "on_session_start", "on_session_end",
        "load_config", "init", "shutdown", "cleanup", "teardown",
        "log_trade", "audit_log", "persist_position",
        "reconnect", "handle_error", "on_reject",
    }

    # Hot path segment tags — used for budget assignment
    SEGMENT_TAGS = {
        "parse":      "market_data_parse",
        "feed":       "market_data_parse",
        "tick":       "market_data_parse",
        "signal":     "signal_eval",
        "alpha":      "signal_eval",
        "strategy":   "signal_eval",
        "risk":       "risk_check",
        "limit":      "risk_check",
        "position":   "risk_check",
        "order":      "order_serialise",
        "submit":     "order_serialise",
        "send":       "order_serialise",
        "dispatch":   "order_serialise",
    }

    def classify(self, unit_name: str, source_snippet: str) -> tuple[PathLabel, str]:
        """
        Returns (PathLabel, unit_tag).
        unit_tag is the hot path segment name for budget mapping.
        """
        # --- Mode 1: annotation-driven ---
        if self.HOT_ANNOTATION.search(source_snippet):
            return PathLabel.HOT, self._infer_tag(unit_name)
        if self.COLD_ANNOTATION.search(source_snippet):
            return PathLabel.COLD, "cold"

        # --- Mode 2: inference-driven ---
        name_lower = unit_name.lower()

        # Direct root match
        if name_lower in self.HOT_ROOTS:
            return PathLabel.HOT, self._infer_tag(unit_name)
        if name_lower in self.COLD_ROOTS:
            return PathLabel.COLD, "cold"

        # Partial name match against root sets
        for root in self.HOT_ROOTS:
            if root in name_lower or name_lower in root:
                return PathLabel.HOT, self._infer_tag(unit_name)
        for root in self.COLD_ROOTS:
            if root in name_lower or name_lower in root:
                return PathLabel.COLD, "cold"

        # Segment keyword match in snippet
        for keyword, tag in self.SEGMENT_TAGS.items():
            if keyword in name_lower:
                return PathLabel.HOT, tag

        # Conservative fallback: unknown → treat as HOT
        return PathLabel.UNKNOWN, self._infer_tag(unit_name)

    def _infer_tag(self, unit_name: str) -> str:
        """Maps function name to a hot path segment tag for budget lookup."""
        name_lower = unit_name.lower()
        for keyword, tag in self.SEGMENT_TAGS.items():
            if keyword in name_lower:
                return tag
        return "signal_eval"   # default hot tag if segment unclear

    def classify_all(self,
                     units: list[tuple[str, str]],
                     latency_budget: LatencyBudget,
                     hw_profile: HardwareProfile
                     ) -> list[CodeUnitContext]:
        """
        Classifies a list of (unit_name, source_snippet) tuples.
        Returns fully populated CodeUnitContext list, budget attached to hot units.
        """
        contexts = []
        for unit_name, snippet in units:
            label, tag = self.classify(unit_name, snippet)
            budget     = latency_budget.budget_for(tag) if label == PathLabel.HOT else 0

            ctx = CodeUnitContext(
                unit_name       = unit_name,
                source_snippet  = snippet,
                path_label      = label,
                unit_tag        = tag,
                budget_ns       = budget,
                hardware_profile= hw_profile,
            )
            contexts.append(ctx)
        return contexts


# ---------------------------------------------------------------------------
# Memory interface (stub — full version in src/memory/experience_store.py)
# ---------------------------------------------------------------------------

class MemoryStub:
    def query_similar(self, embedding: np.ndarray, top_k: int = 5) -> list:
        return []

    def store(self, embedding: np.ndarray, plan: CompilationPlan,
              reward: float, metadata: dict):
        pass


# ---------------------------------------------------------------------------
# Boss Agent — HFT Edition
# ---------------------------------------------------------------------------

class BossAgent:
    """
    Orchestrates the AGentic_C compilation pipeline with HFT-aware routing.

    Responsibilities:
      1. Encode incoming IR into a vector
      2. Query episodic memory for similar past compilations
      3. If hft_mode: classify code units as HOT / COLD
      4. Attach latency budgets to HOT units
      5. Build a CompilationPlan with bifurcated routing
      6. Monitor the agent chain and retry HOT units that fail budget
      7. Store outcomes back to memory after compilation

    Agent chains:
      HOT  path: Fixer(latency) → IR Tuner → HW Tuner → Timing Verifier
      COLD path: Fixer(security) → IR Tuner → HW Tuner  [no budget enforcement]

    Retry logic:
      If Timing Verifier returns estimate > budget:
        → re-route to IR Tuner with tighter directive
        → repeat up to max_retries
        → if still failing: flag to developer, do not emit binary for this unit
    """

    def __init__(self,
                 config_path:  str = "configs/config.yaml",
                 profile_path: str = "configs/hft_profile.yaml"):

        self.config       = load_config(config_path)
        self.boss_cfg     = self.config["agents"]["boss"]
        self.encoder      = SimpleIREncoder()
        self.memory       = MemoryStub()
        self.classifier   = PathClassifier()
        self.hft_profile  = load_hft_profile(profile_path)
        self._log("Boss Agent (HFT Edition) initialized.")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def decide(self, context: CompilationContext) -> CompilationPlan:
        """
        Main entry point.
        Takes a CompilationContext, returns a CompilationPlan.
        Called by pipeline.py before any agent fires.
        """
        self._log(f"Incoming job: {context.source_path} [{context.source_lang}]"
                  f" | hft_mode={context.hft_mode}")

        # Step 1: encode IR if available
        if context.ir_embedding is None:
            ir_text = self._read_ir(context)
            if ir_text:
                context.ir_embedding = self.encoder.encode(ir_text)
                self._log(f"IR encoded → shape {context.ir_embedding.shape}")
            else:
                self._log("No IR yet — will run Pre-IR Fixer first.")

        # Step 2: query memory
        past = []
        if context.ir_embedding is not None:
            past = self.memory.query_similar(
                context.ir_embedding,
                top_k=self.boss_cfg["top_k_memory"]
            )
            if past:
                self._log(f"Memory hit: {len(past)} similar past compilations.")

        # Step 3: HFT — classify code units into HOT / COLD
        if context.hft_mode:
            context = self._classify_units(context)

        # Step 4: build the plan
        plan = self._build_plan(context, past)

        # Step 5: log routing decision
        self._log_plan(plan)

        return plan

    def run_hft_chain(self,
                      plan: CompilationPlan,
                      fixer_agent,
                      ir_tuner_agent,
                      hw_tuner_agent,
                      timing_verifier_agent) -> CompilationPlan:
        """
        Executes the HFT agent chain for all HOT units.
        Handles retry loop: if timing verification fails, re-routes to IR Tuner
        with a tighter directive until budget is met or max_retries exceeded.

        Agents are passed in as callables:
          fixer_agent(unit_ctx)          → modifies unit_ctx in place
          ir_tuner_agent(unit_ctx, plan) → modifies unit_ctx in place
          hw_tuner_agent(unit_ctx, plan) → modifies unit_ctx in place
          timing_verifier_agent(unit_ctx)→ sets unit_ctx.timing_estimate_ns
                                           and unit_ctx.passed_budget
        """
        self._log(f"Starting HFT chain for {len(plan.hot_units)} HOT unit(s).")

        for unit in plan.hot_units:
            self._log(f"  ── HOT unit: {unit.unit_name} "
                      f"[tag={unit.unit_tag}, budget={unit.budget_ns}ns]")

            # Stage 1: Fixer — latency anti-pattern detection
            fixer_agent(unit)
            self._log(f"     Fixer done. Anti-patterns: {unit.anti_patterns or 'none'}")

            # Stage 2-4: IR Tuner → HW Tuner → Timing Verifier (with retry)
            while unit.retry_count <= unit.max_retries:

                # Stage 2: IR Tuner
                ir_tuner_agent(unit, plan)
                self._log(f"     IR Tuner done (retry={unit.retry_count}). "
                           f"{unit.ir_tuner_notes}")

                # Stage 3: HW Tuner
                hw_tuner_agent(unit, plan)

                # Stage 4: Timing Verifier
                timing_verifier_agent(unit)
                self._log(f"     Timing Verifier → estimate={unit.timing_estimate_ns}ns "
                           f"vs budget={unit.budget_ns}ns → "
                           f"{'PASS' if unit.passed_budget else 'FAIL'}")

                if unit.passed_budget:
                    unit.chain_result = ChainResult.PASS
                    break

                # Budget exceeded — tighten and retry
                unit.retry_count += 1
                if unit.retry_count > unit.max_retries:
                    unit.chain_result = ChainResult.FAIL
                    self._log(
                        f"     ⚠ BUDGET EXCEEDED after {unit.max_retries} retries.\n"
                        f"       Unit: {unit.unit_name}\n"
                        f"       Estimate: {unit.timing_estimate_ns}ns  "
                        f"Budget: {unit.budget_ns}ns\n"
                        f"       → Consider algorithmic simplification or "
                        f"reallocating budget in hft_profile.yaml."
                    )
                    break

                # Build tighter directive for next IR Tuner pass
                gap_ns = unit.timing_estimate_ns - unit.budget_ns
                plan.ir_tuner_directive = (
                    f"Unit '{unit.unit_name}' is {gap_ns}ns over budget. "
                    f"Retry {unit.retry_count}/{unit.max_retries}. "
                    f"Focus: branch elimination, cache line compaction, "
                    f"loop unrolling, inline risk checks. "
                    f"Anti-patterns already fixed: {unit.anti_patterns}."
                )
                self._log(f"     Re-routing to IR Tuner: {plan.ir_tuner_directive}")

        # Cold units: run general chain (no timing enforcement)
        self._log(f"Running general chain for {len(plan.cold_units)} COLD unit(s).")
        # Cold chain invocation handled by pipeline.py — Boss just flags them.

        return plan

    def store_outcome(self, context: CompilationContext,
                      plan: CompilationPlan, reward: float):
        """Called by pipeline.py after compilation completes."""
        if context.ir_embedding is not None:
            self.memory.store(
                embedding=context.ir_embedding,
                plan=plan,
                reward=reward,
                metadata={
                    "source_lang":    context.source_lang,
                    "target_arch":    context.target_arch,
                    "source_path":    context.source_path,
                    "hft_mode":       context.hft_mode,
                    "hot_unit_count": len(plan.hot_units),
                    "cold_unit_count":len(plan.cold_units),
                }
            )
            self._log(f"Experience stored (reward={reward:.3f})")

    # ------------------------------------------------------------------
    # Internal — HFT classification
    # ------------------------------------------------------------------

    def _classify_units(self, context: CompilationContext) -> CompilationContext:
        """
        Runs PathClassifier over context.code_units.
        Builds HardwareProfile and LatencyBudget from loaded profile + context.
        Populates context.code_units with fully labelled CodeUnitContext objects.
        """
        # Build HardwareProfile from hft_profile yaml + context
        hw = HardwareProfile(
            cpu            = self.hft_profile.get("hardware", {}).get("cpu", context.target_arch),
            nic            = self.hft_profile.get("hardware", {}).get("nic", "generic"),
            bypass_mode    = self.hft_profile.get("hardware", {}).get("bypass_mode", "none"),
            numa_node      = self.hft_profile.get("hardware", {}).get("numa_node", 0),
            cpu_isolated   = self.hft_profile.get("hardware", {}).get("cpu_isolated", False),
            exchange_proto = self.hft_profile.get("hardware", {}).get("exchange_proto", "fix"),
            arch_string    = context.target_arch,
        )

        # Build LatencyBudget from hft_profile yaml
        lb_cfg = self.hft_profile.get("latency_budget", {})
        lb = LatencyBudget(
            tick_to_order_ns      = lb_cfg.get("tick_to_order_ns",       2000),
            market_data_parse_ns  = lb_cfg.get("market_data_parse_ns",   200),
            signal_eval_ns        = lb_cfg.get("signal_eval_ns",          400),
            risk_check_ns         = lb_cfg.get("risk_check_ns",           150),
            order_serialise_ns    = lb_cfg.get("order_serialise_ns",      250),
            headroom_ns           = lb_cfg.get("headroom_ns",             200),
        )

        context.hardware_profile = hw
        context.latency_budget   = lb

        # Classify each (unit_name, snippet) pair
        raw_units = context.code_units  # expected: list of (str, str) tuples
        if raw_units and isinstance(raw_units[0], tuple):
            context.code_units = self.classifier.classify_all(raw_units, lb, hw)
        # If code_units are already CodeUnitContext objects, leave them.

        hot  = [u for u in context.code_units if u.path_label in (PathLabel.HOT, PathLabel.UNKNOWN)]
        cold = [u for u in context.code_units if u.path_label == PathLabel.COLD]

        self._log(f"Classification complete: {len(hot)} HOT, {len(cold)} COLD units.")
        for u in hot:
            self._log(f"  HOT  → {u.unit_name} [tag={u.unit_tag}, budget={u.budget_ns}ns]")
        for u in cold:
            self._log(f"  COLD → {u.unit_name}")

        return context

    # ------------------------------------------------------------------
    # Internal — plan construction
    # ------------------------------------------------------------------

    def _build_plan(self, context: CompilationContext,
                    past_experiences: list) -> CompilationPlan:
        """
        Core routing logic.
        Priority order:
          1. Memory-informed decision (strong past match)
          2. HFT heuristic rules (if hft_mode)
          3. General heuristic rules
          4. Default: run everything
        """
        plan = CompilationPlan(hw_target=context.target_arch)
        total_budget = context.optimization_budget

        # --- Memory-informed path ---
        if past_experiences:
            best = max(past_experiences, key=lambda x: x.get("reward", 0))
            if best.get("reward", 0) > 0.75:
                past_plan = best.get("plan", {})
                plan.ir_tuner_budget  = past_plan.get("ir_tuner_budget", 25)
                plan.hw_tuner_budget  = past_plan.get("hw_tuner_budget", 10)
                plan.based_on_memory  = True
                plan.confidence       = best["reward"]
                self._log("Memory-informed plan applied.")
                # Still apply HFT unit routing on top of memory plan
                if context.hft_mode:
                    plan = self._apply_hft_routing(plan, context)
                return plan

        # --- Budget allocation ---
        plan.ir_tuner_budget = int(total_budget * 0.67)
        plan.hw_tuner_budget = int(total_budget * 0.33)

        # --- Language-specific adjustments (unchanged) ---
        if context.source_lang == "cpp":
            plan.ir_tuner_budget  += 5
            plan.pre_fixer_focus   = "both"
        elif context.source_lang == "c":
            plan.post_fixer_focus  = "security"

        # --- Architecture-specific adjustments (unchanged) ---
        if "arm64" in context.target_arch:
            plan.hw_target       = "arm64-apple-macosx"
            plan.hw_tuner_budget = max(plan.hw_tuner_budget, 10)

        plan.confidence = 0.5

        # --- HFT routing overlay ---
        if context.hft_mode:
            plan = self._apply_hft_routing(plan, context)

        return plan

    def _apply_hft_routing(self, plan: CompilationPlan,
                            context: CompilationContext) -> CompilationPlan:
        """
        Overlays HFT-specific routing onto an existing plan.
        Bifurcates code units into hot_units and cold_units.
        Activates Timing Verifier for hot path.
        Adjusts Fixer focus to 'latency' for hot units.
        """
        plan.hft_chain_active    = True
        plan.run_timing_verifier = True

        # Hot path: Fixer focuses on latency anti-patterns (not just security)
        plan.pre_fixer_focus  = "latency"
        plan.post_fixer_focus = "latency"

        # Bifurcate units
        plan.hot_units  = [u for u in context.code_units
                           if u.path_label in (PathLabel.HOT, PathLabel.UNKNOWN)]
        plan.cold_units = [u for u in context.code_units
                           if u.path_label == PathLabel.COLD]

        # Boost IR Tuner budget for HFT — hot paths need more passes
        plan.ir_tuner_budget = max(plan.ir_tuner_budget, 35)

        self._log(f"HFT routing active: {len(plan.hot_units)} HOT, "
                  f"{len(plan.cold_units)} COLD units queued.")
        return plan

    # ------------------------------------------------------------------
    # Internal — helpers
    # ------------------------------------------------------------------

    def _read_ir(self, context: CompilationContext) -> Optional[str]:
        ir_dir  = self.config["compiler"]["ir_output_dir"]
        stem    = Path(context.source_path).stem
        ir_path = os.path.join(ir_dir, f"{stem}.ll")
        if os.path.exists(ir_path):
            with open(ir_path, "r") as f:
                return f.read()
        return None

    def _log(self, msg: str):
        print(f"[BossAgent] {msg}")

    def _log_plan(self, plan: CompilationPlan):
        self._log(
            f"Plan decided → "
            f"pre_fixer={plan.run_pre_fixer}({plan.pre_fixer_focus}) | "
            f"post_fixer={plan.run_post_fixer}({plan.post_fixer_focus}) | "
            f"ir_tuner={plan.run_ir_tuner}(budget={plan.ir_tuner_budget}) | "
            f"hw_tuner={plan.run_hw_tuner}(budget={plan.hw_tuner_budget}) | "
            f"timing_verifier={plan.run_timing_verifier} | "
            f"hft_chain={plan.hft_chain_active} | "
            f"hot_units={len(plan.hot_units)} | "
            f"cold_units={len(plan.cold_units)}"
        )


# ---------------------------------------------------------------------------
# Sample hft_profile.yaml content (for reference — write to configs/ dir)
# ---------------------------------------------------------------------------

SAMPLE_HFT_PROFILE = """
# configs/hft_profile.yaml
# HFT latency budget and hardware profile.
# All times in nanoseconds.

latency_budget:
  tick_to_order_ns:       2000   # total hot path budget
  market_data_parse_ns:   200    # parse the wire message
  signal_eval_ns:         400    # evaluate alpha condition
  risk_check_ns:          150    # inline risk gate
  order_serialise_ns:     250    # build + send the order
  headroom_ns:            200    # buffer for jitter and NIC variance

hardware:
  cpu:            x86_64
  nic:            Solarflare_X2
  bypass_mode:    openonload     # none | dpdk | openonload | vma
  numa_node:      0
  cpu_isolated:   true
  exchange_proto: itch           # fix | itch | ouch | sbe
"""


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import tempfile

    print("=" * 70)
    print("AGentic_C — Boss Agent (HFT Edition) Smoke Test")
    print("=" * 70)

    # ── Write minimal configs ──────────────────────────────────────────────
    test_config = {
        "compiler": {
            "frontend":       "clang",
            "target_arch":    "x86_64-linux-gnu",
            "opt_level":      "O0",
            "ir_output_dir":  "/tmp/agentic_c/ir"
        },
        "agents": {
            "boss":     {"top_k_memory": 5, "max_retries": 3},
            "fixer":    {"max_repair_attempts": 3},
            "ir_tuner": {"max_steps": 45, "reward_metric": "IrInstructionCount"},
            "hw_tuner": {"max_steps": 30, "target": "llvm"}
        },
        "ppo":     {"learning_rate": 0.0003, "n_steps": 2048,
                    "batch_size": 64, "n_epochs": 10, "gamma": 0.99},
        "rewards": {"perf_weight": 0.5, "security_weight": 0.35, "size_weight": 0.15},
        "memory":  {"host": "localhost", "port": 5432,
                    "db": "agentic_c", "vector_dim": 256}
    }

    hft_profile = {
        "latency_budget": {
            "tick_to_order_ns":      2000,
            "market_data_parse_ns":  200,
            "signal_eval_ns":        400,
            "risk_check_ns":         150,
            "order_serialise_ns":    250,
            "headroom_ns":           200,
        },
        "hardware": {
            "cpu":            "x86_64",
            "nic":            "Solarflare_X2",
            "bypass_mode":    "openonload",
            "numa_node":      0,
            "cpu_isolated":   True,
            "exchange_proto": "itch",
        }
    }

    tmp_cfg     = tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False)
    tmp_profile = tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False)
    yaml.dump(test_config,  tmp_cfg);     tmp_cfg.close()
    yaml.dump(hft_profile,  tmp_profile); tmp_profile.close()

    # ── Sample code units ─────────────────────────────────────────────────
    sample_units = [
        ("on_market_data",
         "void on_market_data(const Tick& t) { update_book(t); evaluate_signal(); }"),

        ("evaluate_signal",
         "bool evaluate_signal() { return mid_price > ema_fast; }"),

        ("check_risk",
         "bool check_risk(int qty) { return position + qty <= MAX_POSITION; }"),

        ("submit_order",
         "void submit_order(Side s, int qty) { send_fix_message(s, qty); }"),

        ("load_config",
         "void load_config() { cfg = parse_yaml('config.yaml'); }"),

        ("on_disconnect",
         "void on_disconnect() { reconnect_session(); log_event('disconnected'); }"),
    ]

    # ── Sample IR ─────────────────────────────────────────────────────────
    sample_ir = """
    define i32 @evaluate_signal(float %0, float %1) {
      %3 = fcmp ogt float %0, %1
      %4 = zext i1 %3 to i32
      ret i32 %4
    }
    define void @on_market_data(ptr %0) {
      call void @update_book(ptr %0)
      call i32 @evaluate_signal(float 0.0, float 0.0)
      ret void
    }
    """

    encoder   = SimpleIREncoder()
    embedding = encoder.encode(sample_ir)
    print(f"\n✓ IR encoded: shape={embedding.shape}, "
          f"non-zero dims={np.count_nonzero(embedding)}")

    # ── Build context ──────────────────────────────────────────────────────
    ctx = CompilationContext(
        source_path        = "/tmp/strategy.cpp",
        source_lang        = "cpp",
        target_arch        = "x86_64-linux-gnu",
        ir_embedding       = embedding,
        optimization_budget= 45,
        hft_mode           = True,
        code_units         = sample_units,   # list of (name, snippet) tuples
    )

    # ── Run Boss Agent ─────────────────────────────────────────────────────
    agent = BossAgent(config_path=tmp_cfg.name, profile_path=tmp_profile.name)
    plan  = agent.decide(ctx)

    print(f"\n✓ Plan generated:")
    print(f"  hft_chain_active  : {plan.hft_chain_active}")
    print(f"  hot_units         : {[u.unit_name for u in plan.hot_units]}")
    print(f"  cold_units        : {[u.unit_name for u in plan.cold_units]}")
    print(f"  timing_verifier   : {plan.run_timing_verifier}")
    print(f"  ir_tuner_budget   : {plan.ir_tuner_budget}")
    print(f"  confidence        : {plan.confidence}")

    # ── Simulate HFT chain with stub agents ───────────────────────────────
    print("\n── Simulating HFT chain with stub agents ──")

    def stub_fixer(unit: CodeUnitContext):
        # Detect obvious latency anti-patterns in snippet
        if "new " in unit.source_snippet or "malloc" in unit.source_snippet:
            unit.anti_patterns.append("heap_alloc_on_hot_path")
        if "virtual" in unit.source_snippet:
            unit.anti_patterns.append("virtual_dispatch")
        unit.fixer_notes = f"Fixer pass complete. Found: {unit.anti_patterns or 'none'}"

    def stub_ir_tuner(unit: CodeUnitContext, plan: CompilationPlan):
        unit.ir_optimised   = True
        directive           = plan.ir_tuner_directive or "standard HFT passes"
        unit.ir_tuner_notes = f"Applied: branch-free arith, cache layout. Directive: '{directive}'"

    def stub_hw_tuner(unit: CodeUnitContext, plan: CompilationPlan):
        pass   # sets hw-specific flags — stub does nothing

    # Timing verifier: simulate pass on first try for most, fail once for submit_order
    _submit_attempts = {"n": 0}
    def stub_timing_verifier(unit: CodeUnitContext):
        if unit.unit_name == "submit_order" and _submit_attempts["n"] == 0:
            _submit_attempts["n"] += 1
            unit.timing_estimate_ns = unit.budget_ns + 80   # over budget: triggers retry
            unit.passed_budget      = False
        else:
            # Simulate finishing within budget
            unit.timing_estimate_ns = int(unit.budget_ns * 0.85)
            unit.passed_budget      = True

    plan = agent.run_hft_chain(
        plan,
        fixer_agent           = stub_fixer,
        ir_tuner_agent        = stub_ir_tuner,
        hw_tuner_agent        = stub_hw_tuner,
        timing_verifier_agent = stub_timing_verifier,
    )

    print(f"\n✓ HFT chain results:")
    for u in plan.hot_units:
        print(f"  {u.unit_name:<22} → {u.chain_result.value.upper():<6} "
              f"({u.timing_estimate_ns}ns / {u.budget_ns}ns budget) "
              f"retries={u.retry_count}")

    # ── Store outcome ──────────────────────────────────────────────────────
    agent.store_outcome(ctx, plan, reward=0.88)

    print("\n✓ Boss Agent (HFT Edition) smoke test PASSED")
    print("=" * 70)