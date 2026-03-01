"""
AGentic_C — Pipeline (HFT Edition)
=====================================
Wires all agents into a single end-to-end compilation pipeline.

Usage:
    from src.pipeline import Pipeline
    result = Pipeline().compile("src/strategy.cpp")

Or from CLI:
    python3 src/pipeline.py src/strategy.cpp

Pipeline stages:
    1. Clang frontend   → emit LLVM IR (.ll)
    2. Boss Agent       → classify HOT/COLD, build plan
    3. Fixer Agent      → syntax repair + HFT anti-pattern scan
    4. IR Tuner         → algorithm-level LLVM pass optimisation
    5. HW Tuner         → ISA-specific + NEON optimisation
    6. Timing Verifier  → latency budget verdict (inside HW Tuner)
    7. Boss retry loop  → up to max_retries if budget not met
    8. Experience Store → save (embedding, plan, reward) to pgvector
    9. Emit binary      → clang final codegen from optimised IR

HOT units go through stages 3-7 with latency enforcement.
COLD units go through stages 3-4 (general optimisation only).

All config read from configs/config.yaml.
"""

import os
import re
import sys
import time
import yaml
import json
import shutil
import tempfile
import subprocess
import numpy as np
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

# ---------------------------------------------------------------------------
# Agent imports
# Each agent is designed to be independently importable.
# ---------------------------------------------------------------------------

# Resolve paths whether run from project root or src/
_SRC = Path(__file__).parent
_ROOT = _SRC.parent if _SRC.name == "src" else _SRC
sys.path.insert(0, str(_ROOT / "src" / "agents"))
sys.path.insert(0, str(_ROOT / "src"))

try:
    from agents.boss_agent      import BossAgent, CompilationContext, PathLabel
    from agents.fixer_agent_hft import FixerAgent
    from agents.ir_tuner_agent  import IRTunerAgent
    from agents.hw_tuner_agent  import HWTunerAgent
except ImportError:
    # Try flat import (running from src/agents/ directly)
    from boss_agent      import BossAgent, CompilationContext, PathLabel
    from fixer_agent_hft import FixerAgent
    from ir_tuner_agent  import IRTunerAgent
    from hw_tuner_agent  import HWTunerAgent


# ---------------------------------------------------------------------------
# Pipeline result
# ---------------------------------------------------------------------------

@dataclass
class PipelineResult:
    success:          bool
    source_path:      str
    binary_path:      str             = ""
    ir_path:          str             = ""   # final optimised IR

    # Timing
    total_time_s:     float           = 0.0
    stage_times:      dict            = field(default_factory=dict)

    # Per-unit results
    hot_unit_results: list            = field(default_factory=list)
    cold_unit_results: list           = field(default_factory=list)

    # Summary metrics
    total_hot_units:      int         = 0
    hot_units_passed:     int         = 0
    hot_units_failed:     int         = 0
    total_retries:        int         = 0
    avg_latency_reduction: float      = 0.0   # % improvement across hot units
    experience_stored:    bool        = False
    reward:               float       = 0.0

    # Plan and config snapshot
    hft_mode:         bool            = False
    config_snapshot:  dict            = field(default_factory=dict)
    notes:            str             = ""


@dataclass
class UnitResult:
    unit_name:        str
    path_label:       str             # 'hot' | 'cold'
    budget_ns:        int             = 0
    anti_patterns:    list            = field(default_factory=list)
    latency_before_ns: float          = 0.0
    latency_after_ns:  float          = 0.0
    within_budget:    bool            = True
    retries:          int             = 0
    passes_applied:   list            = field(default_factory=list)
    verdict:          str             = "PASS"   # 'PASS' | 'FAIL' | 'ADVISORY'
    notes:            str             = ""


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

class Pipeline:
    """
    End-to-end compilation pipeline for AGentic_C.

    Instantiate once, call compile() for each source file.
    Agents are initialised once and reused across calls.
    """

    def __init__(self,
                 config_path: str  = "configs/config.yaml",
                 stub_mode:   bool = True,
                 verbose:     bool = True):
        self.config_path = config_path
        self.stub_mode   = stub_mode
        self.verbose     = verbose
        self.config      = self._load_config(config_path)

        # Initialise all agents once
        self._log("Initialising pipeline agents...")
        t0 = time.perf_counter()

        self.boss_agent = BossAgent(config_path=config_path)
        self.fixer      = FixerAgent()
        self.ir_tuner   = IRTunerAgent(config_path=config_path, stub_mode=stub_mode)
        self.hw_tuner   = HWTunerAgent(config_path=config_path, stub_mode=stub_mode)

        init_ms = (time.perf_counter() - t0) * 1000
        self._log(f"All agents ready in {init_ms:.0f}ms.")

        self.hft_mode  = self.config.get("pipeline", {}).get("hft_mode", True)
        self.max_retries = self.config.get("agents", {}) \
                               .get("boss", {}).get("max_retries", 3)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def compile(self, source_path: str,
                emit_binary: bool = False) -> PipelineResult:
        """
        Compile a C/C++ source file through the full agent pipeline.

        Args:
            source_path:  path to .cpp or .c source file
            emit_binary:  if True, invoke clang to produce final binary
                          (requires clang installed; False = IR only)

        Returns:
            PipelineResult with all metrics and per-unit results
        """
        t_start = time.perf_counter()
        self._log("=" * 68)
        self._log(f"AGentic_C Pipeline — {Path(source_path).name}")
        self._log(f"mode={'HFT' if self.hft_mode else 'general'}")
        self._log("=" * 68)

        stage_times = {}
        result = PipelineResult(
            success      = False,
            source_path  = source_path,
            hft_mode     = self.hft_mode,
            config_snapshot = {
                "hft_mode":   self.hft_mode,
                "max_retries": self.max_retries,
            }
        )

        # ── Stage 1: Clang frontend → IR ──────────────────────────────
        t = time.perf_counter()
        ir_path = self._emit_ir(source_path)
        stage_times["clang_frontend"] = time.perf_counter() - t

        if not ir_path:
            result.notes = "Clang frontend failed to emit IR."
            return result
        self._log(f"[1/6] IR emitted: {ir_path}")

        # ── Stage 2: Boss Agent — classify + plan ──────────────────────
        t = time.perf_counter()

        # Pre-populate code_units by parsing source for function names
        code_units = self._extract_code_units(source_path, ir_path)

        ctx = CompilationContext(
            source_path  = source_path,
            source_lang  = Path(source_path).suffix.lstrip("."),
            target_arch  = self.config.get("compiler", {}).get("target_arch", "arm64"),
            ir_embedding = None,
            hft_mode     = self.hft_mode,
            code_units   = code_units,
        )
        plan = self.boss_agent.decide(ctx)
        stage_times["boss_agent"] = time.perf_counter() - t
        self._log(f"[2/6] Plan: {len(plan.hot_units)} HOT, "
                  f"{len(plan.cold_units)} COLD units | "
                  f"ir_tuner_budget={plan.ir_tuner_budget} steps")

        # ── Stage 3-6: Per-unit agent chain ───────────────────────────
        t = time.perf_counter()
        hot_results, cold_results = self._run_agent_chains(
            plan, ir_path, stage_times
        )
        stage_times["agent_chains"] = time.perf_counter() - t

        # ── Stage 7: Emit binary (optional) ───────────────────────────
        binary_path = ""
        if emit_binary:
            t = time.perf_counter()
            binary_path = self._emit_binary(ir_path, source_path)
            stage_times["codegen"] = time.perf_counter() - t
            if binary_path:
                self._log(f"[6/6] Binary: {binary_path}")

        # ── Stage 8: Experience store ──────────────────────────────────
        t = time.perf_counter()
        reward = self._compute_pipeline_reward(hot_results, cold_results)
        stored = self._store_experience(plan, reward)
        stage_times["experience_store"] = time.perf_counter() - t

        # ── Assemble result ────────────────────────────────────────────
        total_s = time.perf_counter() - t_start

        passed  = [r for r in hot_results if r.verdict == "PASS"]
        failed  = [r for r in hot_results if r.verdict == "FAIL"]
        retries = sum(r.retries for r in hot_results)

        lat_reductions = [
            (r.latency_before_ns - r.latency_after_ns) / max(r.latency_before_ns, 1)
            for r in hot_results if r.latency_before_ns > 0
        ]
        avg_reduction = np.mean(lat_reductions) if lat_reductions else 0.0

        result.success              = len(failed) == 0
        result.ir_path              = ir_path
        result.binary_path          = binary_path
        result.total_time_s         = round(total_s, 3)
        result.stage_times          = {k: round(v*1000, 1) for k, v in stage_times.items()}
        result.hot_unit_results     = hot_results
        result.cold_unit_results    = cold_results
        result.total_hot_units      = len(hot_results)
        result.hot_units_passed     = len(passed)
        result.hot_units_failed     = len(failed)
        result.total_retries        = retries
        result.avg_latency_reduction= round(avg_reduction * 100, 1)
        result.experience_stored    = stored
        result.reward               = round(reward, 4)
        result.notes = (
            f"{len(passed)}/{len(hot_results)} HOT units within budget. "
            f"{retries} retries. "
            f"Avg latency reduction: {result.avg_latency_reduction:.1f}%."
        )

        self._print_summary(result)
        return result

    # ------------------------------------------------------------------
    # Agent chain execution
    # ------------------------------------------------------------------

    def _run_agent_chains(self, plan, ir_path: str,
                          stage_times: dict) -> tuple[list, list]:
        """
        Runs Fixer → IR Tuner → HW Tuner for each code unit.
        HOT units get full HFT chain with budget enforcement.
        COLD units get general optimisation only.
        """
        hot_results  = []
        cold_results = []

        # ── HOT units ─────────────────────────────────────────────────
        if plan.hot_units:
            self._log(f"[3/6] Running HFT chain for {len(plan.hot_units)} HOT unit(s)...")
            for unit in plan.hot_units:
                r = self._run_hot_unit(unit, ir_path, plan)
                hot_results.append(r)

        # ── COLD units ────────────────────────────────────────────────
        if plan.cold_units:
            self._log(f"[5/6] Running general chain for "
                      f"{len(plan.cold_units)} COLD unit(s)...")
            for unit in plan.cold_units:
                r = self._run_cold_unit(unit, ir_path, plan)
                cold_results.append(r)

        return hot_results, cold_results

    def _run_hot_unit(self, unit, ir_path: str, plan) -> UnitResult:
        """
        Full HFT chain for one HOT unit:
          Fixer (anti-pattern scan) → IR Tuner → HW Tuner → Timing Verifier
          → retry loop if budget not met
        """
        unit_name  = getattr(unit, "unit_name", str(unit))
        budget_ns  = getattr(unit, "budget_ns", 0)
        retry_count= 0
        directive  = "standard HFT passes"

        self._log(f"  ── HOT: {unit_name} [budget={budget_ns}ns]")

        # Stage A: Fixer — HFT anti-pattern scan
        anti_patterns = []
        fixer_notes   = ""
        if hasattr(unit, "source_snippet") and unit.source_snippet:
            fixer_result = self.fixer.hft_fix(
                unit.source_snippet, unit_name, "hot"
            )
            anti_patterns = [
                f"{ap.code}:{ap.severity.value}:{ap.line_hint}"
                for ap in fixer_result.anti_patterns
            ]
            fixer_notes = fixer_result.message
            if anti_patterns:
                self._log(f"     Fixer: {len(anti_patterns)} anti-pattern(s): "
                          f"{[ap.split(':')[0] for ap in anti_patterns]}")
            else:
                self._log(f"     Fixer: clean")
        else:
            self._log(f"     Fixer: no source snippet — skipping scan")

        # Stage B+C: IR Tuner + HW Tuner with retry loop
        latency_before = 0.0
        latency_after  = 0.0
        passes_applied = []
        within_budget  = False

        for attempt in range(self.max_retries + 1):
            # IR Tuner
            ir_result = self.ir_tuner.tune(
                ir_path       = ir_path,
                budget_steps  = plan.ir_tuner_budget,
                hft_mode      = True,
                budget_ns     = budget_ns,
                anti_patterns = anti_patterns,
                directive     = directive,
            )
            if attempt == 0:
                latency_before = ir_result.latency_before_ns

            # HW Tuner
            hw_result = self.hw_tuner.tune(
                ir_path       = ir_result.ir_path_out or ir_path,
                budget_steps  = plan.hw_tuner_budget,
                hft_mode      = True,
                budget_ns     = budget_ns,
                anti_patterns = anti_patterns,
                directive     = directive,
            )

            latency_after  = hw_result.latency_after_ns
            passes_applied = ir_result.passes_applied + hw_result.passes_applied
            within_budget  = hw_result.within_budget

            verdict_str = "✓ PASS" if within_budget else "✗ FAIL"
            self._log(f"     [{attempt+1}/{self.max_retries+1}] "
                      f"Latency: {latency_after:.0f}ns vs {budget_ns}ns → {verdict_str}")

            if within_budget:
                break

            # Retry — tighten directive from Timing Verifier
            if hw_result.timing_verdict and hw_result.timing_verdict.directive:
                directive = hw_result.timing_verdict.directive
            else:
                directive = (
                    f"Retry {attempt+1}: still over budget. "
                    f"Aggressive inlining, loop unrolling, eliminate branches."
                )
            retry_count += 1

        verdict = "PASS" if within_budget else "FAIL"

        return UnitResult(
            unit_name         = unit_name,
            path_label        = "hot",
            budget_ns         = budget_ns,
            anti_patterns     = anti_patterns,
            latency_before_ns = latency_before,
            latency_after_ns  = latency_after,
            within_budget     = within_budget,
            retries           = retry_count,
            passes_applied    = passes_applied,
            verdict           = verdict,
            notes             = fixer_notes,
        )

    def _run_cold_unit(self, unit, ir_path: str, plan) -> UnitResult:
        """
        General chain for COLD units — no budget enforcement.
        IR Tuner only (HW Tuner optional for cold path).
        """
        unit_name = getattr(unit, "unit_name", str(unit))
        self._log(f"  ── COLD: {unit_name}")

        ir_result = self.ir_tuner.tune(
            ir_path       = ir_path,
            budget_steps  = plan.ir_tuner_budget,
            hft_mode      = False,
        )

        return UnitResult(
            unit_name         = unit_name,
            path_label        = "cold",
            budget_ns         = 0,
            latency_before_ns = ir_result.latency_before_ns,
            latency_after_ns  = ir_result.latency_after_ns,
            within_budget     = True,
            passes_applied    = ir_result.passes_applied,
            verdict           = "ADVISORY",
            notes             = ir_result.notes,
        )

    # ------------------------------------------------------------------
    # Clang integration
    # ------------------------------------------------------------------

    def _extract_code_units(self, source_path: str, ir_path: str) -> list:
        """
        Extracts function names and snippets from source or IR.
        Creates CodeUnitContext objects for the Boss Agent to classify.
        Falls back to IR function names if source is unavailable.
        """
        try:
            from agents.boss_agent import CodeUnitContext, PathLabel
        except ImportError:
            from boss_agent import CodeUnitContext, PathLabel

        units = []
        source_text = ""

        # Try reading source file
        if os.path.exists(source_path):
            with open(source_path) as f:
                source_text = f.read()

        # Extract function signatures from source using regex
        # Handles: void foo(...), int bar(...), [[hft::hot]] void baz(...)
        fn_pattern = re.compile(
            r'(?:\[\[hft::(?:hot|cold)\]\]\s*)?'     # optional HFT annotation
            r'(?:inline\s+|static\s+|virtual\s+)*'   # optional qualifiers
            r'[\w:<>*&]+\s+'                          # return type
            r'(\w+)\s*\([^)]*\)\s*'                  # function name + params
            r'(?:noexcept\s*)?(?:const\s*)?'
            r'\{',                                    # opening brace
            re.MULTILINE
        )

        # Also check for [[hft::hot]] / [[hft::cold]] annotations
        hot_annot  = re.compile(r'\[\[hft::hot\]\]')
        cold_annot = re.compile(r'\[\[hft::cold\]\]')

        if source_text:
            # Split source into function blocks
            lines  = source_text.split("\n")
            blocks = {}   # fn_name → snippet

            for i, line in enumerate(lines):
                m = fn_pattern.search(line)
                if m:
                    fn_name = m.group(1)
                    # Grab up to 20 lines as the snippet
                    snippet = "\n".join(lines[max(0, i-1):i+20])
                    blocks[fn_name] = snippet

            for fn_name, snippet in blocks.items():
                units.append(CodeUnitContext(
                    unit_name      = fn_name,
                    source_snippet = snippet,
                    path_label     = PathLabel.UNKNOWN,
                ))

        # Fallback: extract function names from IR
        if not units and os.path.exists(ir_path):
            with open(ir_path) as f:
                ir_text = f.read()
            for m in re.finditer(r'^define\s+\S+\s+@(\w+)\s*\(', ir_text, re.MULTILINE):
                fn_name = m.group(1)
                units.append(CodeUnitContext(
                    unit_name      = fn_name,
                    source_snippet = f"; IR function @{fn_name}",
                    path_label     = PathLabel.UNKNOWN,
                ))

        self._log(f"  Extracted {len(units)} code unit(s): "
                  f"{[u.unit_name for u in units]}")
        return units

    def _emit_ir(self, source_path: str) -> str:
        """
        Runs clang to emit LLVM IR (.ll file).
        Falls back to stub IR if clang is not available.
        """
        ir_dir = self.config.get("compiler", {}).get("ir_output_dir", "/tmp/agentic_c/ir")
        os.makedirs(ir_dir, exist_ok=True)

        stem    = Path(source_path).stem
        ir_path = os.path.join(ir_dir, f"{stem}.ll")

        # Try real clang
        if shutil.which("clang") and os.path.exists(source_path):
            arch   = self.config.get("compiler", {}).get("target_arch", "")
            target = [f"--target={arch}"] if arch else []
            cmd = ["clang", "-S", "-emit-llvm", "-O0", "-Xclang",
                   "-disable-O0-optnone"] + target + [source_path, "-o", ir_path]
            try:
                result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
                if result.returncode == 0 and os.path.exists(ir_path):
                    return ir_path
                self._log(f"  Clang error: {result.stderr[:120]}")
            except Exception as e:
                self._log(f"  Clang failed: {e}")

        # Fallback — stub IR for testing without clang
        stub = _STUB_IR_TEMPLATE.format(stem=stem)
        with open(ir_path, "w") as f:
            f.write(stub)
        self._log(f"  (Stub IR — clang not available or source not found)")
        return ir_path

    def _emit_binary(self, ir_path: str, source_path: str) -> str:
        """
        Final codegen: clang takes optimised IR → native binary.
        """
        if not shutil.which("clang"):
            self._log("  codegen skipped — clang not available")
            return ""

        out_path = ir_path.replace(".ll", "").replace("_opt", "") + "_bin"
        cmd = ["clang", ir_path, "-o", out_path]
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
            if result.returncode == 0:
                return out_path
            self._log(f"  codegen error: {result.stderr[:120]}")
        except Exception as e:
            self._log(f"  codegen failed: {e}")
        return ""

    # ------------------------------------------------------------------
    # Reward and experience store
    # ------------------------------------------------------------------

    def _compute_pipeline_reward(self, hot_results: list,
                                 cold_results: list) -> float:
        """
        Composite pipeline reward for the experience store.
        Weighted average of per-unit outcomes.
        """
        if not hot_results and not cold_results:
            return 0.5

        scores = []
        for r in hot_results:
            # HOT unit score: latency improvement + budget hit
            if r.latency_before_ns > 0:
                lat_improvement = max(0.0, (r.latency_before_ns - r.latency_after_ns)
                                          / r.latency_before_ns)
            else:
                lat_improvement = 0.0
            budget_bonus = 0.2 if r.within_budget else 0.0
            retry_penalty = 0.05 * r.retries
            scores.append(min(1.0, lat_improvement + budget_bonus - retry_penalty))

        for r in cold_results:
            # COLD unit: just instruction reduction, no budget pressure
            if r.latency_before_ns > 0:
                scores.append(max(0.0, (r.latency_before_ns - r.latency_after_ns)
                                       / r.latency_before_ns))
            else:
                scores.append(0.5)

        return round(float(np.mean(scores)), 4) if scores else 0.5

    def _store_experience(self, plan, reward: float) -> bool:
        """
        Saves (embedding, plan, reward) to the experience store.
        Stubs gracefully if PostgreSQL / pgvector is not available.
        """
        min_threshold = self.config.get("memory", {}).get(
                        "min_reward_threshold", 0.75)

        if reward < min_threshold:
            self._log(f"[8/8] Experience NOT stored "
                      f"(reward={reward:.3f} < threshold={min_threshold})")
            return False

        try:
            # Try real pgvector store (from src/memory/)
            from memory.experience_store import ExperienceStore
            store = ExperienceStore(config=self.config)
            store.save(plan=plan, reward=reward)
            self._log(f"[8/8] Experience stored (reward={reward:.3f})")
            return True
        except ImportError:
            # Stub — log only
            self._log(f"[8/8] Experience stored (stub, reward={reward:.3f})")
            return True
        except Exception as e:
            self._log(f"[8/8] Experience store failed: {e}")
            return False

    # ------------------------------------------------------------------
    # Logging and helpers
    # ------------------------------------------------------------------

    def _print_summary(self, result: PipelineResult):
        self._log("=" * 68)
        self._log("Pipeline Summary")
        self._log("=" * 68)
        self._log(f"  Status:     {'✓ SUCCESS' if result.success else '✗ FAILED'}")
        self._log(f"  HOT units:  {result.hot_units_passed}/{result.total_hot_units} passed")
        if result.hot_units_failed > 0:
            self._log(f"  ⚠ FAILED:  {result.hot_units_failed} unit(s) over latency budget")
        self._log(f"  Retries:    {result.total_retries}")
        self._log(f"  Avg Δlat:   {result.avg_latency_reduction:.1f}%")
        self._log(f"  Reward:     {result.reward:.4f}")
        self._log(f"  Wall time:  {result.total_time_s*1000:.0f}ms")
        self._log(f"  Stages (ms): {result.stage_times}")
        self._log("=" * 68)

        # Per-unit table
        if result.hot_unit_results:
            self._log("\n  HOT unit results:")
            for r in result.hot_unit_results:
                icon = "✓" if r.verdict == "PASS" else "✗"
                lat  = f"{r.latency_after_ns:.0f}ns / {r.budget_ns}ns"
                aps  = [a.split(":")[0] for a in r.anti_patterns]
                self._log(
                    f"    {icon} {r.unit_name:<25} {lat:<18} "
                    f"retries={r.retries}  AP={aps if aps else 'clean'}"
                )

        if result.cold_unit_results:
            self._log("\n  COLD unit results:")
            for r in result.cold_unit_results:
                self._log(f"    ~ {r.unit_name:<25} advisory  "
                          f"passes={len(r.passes_applied)}")

    def _load_config(self, path: str) -> dict:
        if os.path.exists(path):
            with open(path) as f:
                return yaml.safe_load(f)
        return {}

    def _log(self, msg: str):
        if self.verbose:
            print(msg)


# ---------------------------------------------------------------------------
# Stub IR template (used when clang is not available)
# ---------------------------------------------------------------------------

_STUB_IR_TEMPLATE = """
; AGentic_C stub IR for {stem}
; Generated because clang was unavailable or source file not found.

define float @on_market_data(float %price, float %ema) {{
entry:
  %p = alloca float, align 4
  %e = alloca float, align 4
  store float %price, float* %p, align 4
  store float %ema,   float* %e, align 4
  %pv = load float, float* %p, align 4
  %ev = load float, float* %e, align 4
  %diff = fsub float %pv, %ev
  %cmp  = fcmp ogt float %diff, 0.0
  %res  = select i1 %cmp, float %diff, float 0.0
  ret float %res
}}

define i32 @check_risk(i32 %qty, i32 %pos) {{
entry:
  %sum = add i32 %qty, %pos
  %ok  = icmp slt i32 %sum, 1000
  %r   = zext i1 %ok to i32
  ret i32 %r
}}

define i32 @evaluate_signal(float %fast, float %slow) {{
entry:
  %diff = fsub float %fast, %slow
  %cmp  = fcmp ogt float %diff, 0.0
  %r    = zext i1 %cmp to i32
  ret i32 %r
}}

define void @load_config() {{
entry:
  ret void
}}
"""


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main():
    import argparse
    parser = argparse.ArgumentParser(description="AGentic_C HFT Compiler Pipeline")
    parser.add_argument("source", nargs="?", default="",
                        help="C/C++ source file to compile")
    parser.add_argument("--config", default="configs/config.yaml",
                        help="Path to config.yaml")
    parser.add_argument("--binary", action="store_true",
                        help="Emit native binary from optimised IR")
    parser.add_argument("--no-stub", action="store_true",
                        help="Disable stub mode (requires real CompilerGym)")
    args = parser.parse_args()

    stub_mode = not args.no_stub

    # If no source given, run smoke test
    if not args.source:
        run_smoke_test(config_path=args.config, stub_mode=stub_mode)
        return

    pipeline = Pipeline(config_path=args.config, stub_mode=stub_mode)
    result   = pipeline.compile(args.source, emit_binary=args.binary)

    sys.exit(0 if result.success else 1)


def run_smoke_test(config_path: str = "configs/config.yaml",
                   stub_mode: bool = True):
    """
    End-to-end smoke test with a fake strategy.cpp source.
    Verifies the whole pipeline runs without errors.
    """
    print("=" * 68)
    print("AGentic_C Pipeline — End-to-End Smoke Test")
    print("=" * 68)

    # Create a fake strategy.cpp for testing
    strategy_cpp = """
// HFT Strategy — sample source for pipeline smoke test
#include <cstdint>

[[hft::hot]]
float on_market_data(float price, float ema_fast) {
    float diff = price - ema_fast;
    return diff > 0.0f ? diff : 0.0f;
}

[[hft::hot]]
int check_risk(int qty, int position) {
    return (qty + position) < 1000;
}

[[hft::hot]]
float evaluate_signal(float fast, float slow) {
    return fast - slow;
}

[[hft::cold]]
void load_config() {
    // startup only
}
"""
    tmp = tempfile.NamedTemporaryFile(suffix=".cpp", mode="w",
                                     delete=False, dir="/tmp",
                                     prefix="strategy_")
    tmp.write(strategy_cpp)
    tmp.close()
    source_path = tmp.name
    print(f"\n✓ Stub strategy.cpp written: {source_path}\n")

    pipeline = Pipeline(config_path=config_path, stub_mode=stub_mode)
    result   = pipeline.compile(source_path, emit_binary=False)

    print("\n── Assertions ──")

    # 1. Pipeline completed
    assert result.source_path == source_path, "source_path mismatch"
    print("  ✓ Pipeline completed without exception")

    # 2. IR was emitted
    assert result.ir_path and os.path.exists(result.ir_path), \
        f"IR file missing: {result.ir_path}"
    print(f"  ✓ IR file emitted: {result.ir_path}")

    # 3. Hot units were classified
    assert result.total_hot_units > 0, "No HOT units classified"
    print(f"  ✓ HOT units: {result.total_hot_units}")

    # 4. All hot units have verdicts
    for r in result.hot_unit_results:
        assert r.verdict in ("PASS", "FAIL"), f"Unexpected verdict: {r.verdict}"
    print(f"  ✓ All HOT unit verdicts set: "
          f"{[r.verdict for r in result.hot_unit_results]}")

    # 5. Cold units (advisory — depends on source)
    cold_count = len(result.cold_unit_results)
    print(f"  ✓ COLD units: {cold_count} "
          f"{'(some cold units detected)' if cold_count else '(all classified HOT by Boss Agent)'}")

    # 6. Timing present
    assert result.total_time_s > 0, "No timing recorded"
    print(f"  ✓ Wall time: {result.total_time_s*1000:.0f}ms")

    # 7. Stage times recorded
    assert "boss_agent" in result.stage_times, "Boss agent time missing"
    print(f"  ✓ Stage times: {result.stage_times}")

    # 8. Reward in range
    assert 0.0 <= result.reward <= 1.0, f"Reward out of range: {result.reward}"
    print(f"  ✓ Reward: {result.reward:.4f}")

    # 9. Notes populated
    assert result.notes, "Notes empty"
    print(f"  ✓ Notes: {result.notes}")

    os.unlink(source_path)

    print()
    print("=" * 68)
    status = "✓ PASSED" if result.success else "~ COMPLETED (some units over budget)"
    print(f"Pipeline smoke test {status}")
    print("=" * 68)

    return result


if __name__ == "__main__":
    main()
