"""
AGentic_C — Boss Agent
Orchestrates the full compilation pipeline.
Decides which agents fire, in what order, with what budget.
"""

import os
import yaml
import numpy as np
from dataclasses import dataclass, field
from typing import Optional
from pathlib import Path


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class CompilationContext:
    """Everything Boss Agent knows about an incoming compilation job."""
    source_path: str
    source_lang: str                        # 'c', 'cpp', 'rust'
    target_arch: str                        # 'arm64', 'x86_64'
    ir_embedding: Optional[np.ndarray]      # 256-dim float vector (set after encoding)
    optimization_budget: int = 45          # max steps across all agents
    past_experiences: list = field(default_factory=list)  # from pgvector


@dataclass
class CompilationPlan:
    """Structured decision output from Boss Agent."""
    # Pre-IR Fixer
    run_pre_fixer: bool = True
    pre_fixer_focus: str = "syntax"         # 'syntax' | 'security' | 'both'

    # Post-IR Fixer
    run_post_fixer: bool = True
    post_fixer_focus: str = "security"

    # IR Tuner
    run_ir_tuner: bool = True
    ir_tuner_budget: int = 30              # max pass-application steps

    # HW Tuner
    run_hw_tuner: bool = True
    hw_tuner_budget: int = 15
    hw_target: str = "arm64-apple-macosx"

    # Routing metadata
    confidence: float = 0.0               # boss confidence in this plan
    based_on_memory: bool = False         # did memory inform this decision?
    retry_count: int = 0


# ---------------------------------------------------------------------------
# Config loader
# ---------------------------------------------------------------------------

def load_config(config_path: str = "configs/config.yaml") -> dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


# ---------------------------------------------------------------------------
# IR Encoder (lightweight — full version lives in src/encoder/ir_encoder.py)
# ---------------------------------------------------------------------------

class SimpleIREncoder:
    """
    Encodes LLVM IR text into a fixed-size float vector.
    
    This is the lightweight version using handcrafted Autophase-style features.
    The full version (in encoder/ir_encoder.py) uses a GNN over the CFG.
    We use this for the Boss Agent so it has zero deep learning overhead.
    """

    DIM = 256

    # LLVM IR instruction keywords we count as features
    INSTRUCTION_TYPES = [
        "alloca", "load", "store", "add", "sub", "mul", "sdiv", "udiv",
        "fadd", "fsub", "fmul", "fdiv", "icmp", "fcmp", "br", "ret",
        "call", "phi", "select", "getelementptr", "bitcast", "zext",
        "sext", "trunc", "and", "or", "xor", "shl", "lshr", "ashr",
        "switch", "invoke", "unreachable", "extractvalue", "insertvalue",
    ]

    def encode(self, ir_text: str) -> np.ndarray:
        """
        Produces a 256-dim float32 vector from raw LLVM IR text.
        
        First 35 dims  → normalized instruction type counts
        Next  10 dims  → structural features (functions, blocks, etc.)
        Rest  211 dims → zero-padded (reserved for GNN encoder upgrade)
        """
        vec = np.zeros(self.DIM, dtype=np.float32)
        lines = ir_text.lower().split("\n")
        total_instructions = max(len(lines), 1)

        # Feature block 1: instruction type distribution (dims 0-34)
        for i, instr in enumerate(self.INSTRUCTION_TYPES):
            count = sum(1 for line in lines if instr in line)
            vec[i] = count / total_instructions  # normalize

        # Feature block 2: structural features (dims 35-44)
        vec[35] = ir_text.count("define ")    / 100.0   # function count
        vec[36] = ir_text.count("declare ")   / 100.0   # extern decls
        vec[37] = total_instructions          / 1000.0  # total lines
        vec[38] = ir_text.count("phi")        / max(ir_text.count("define "), 1)  # phi density
        vec[39] = ir_text.count("call")       / total_instructions   # call density
        vec[40] = ir_text.count("br")         / total_instructions   # branch density
        vec[41] = ir_text.count("loop")       / total_instructions   # loop hints
        vec[42] = ir_text.count("ptr")        / total_instructions   # pointer density
        vec[43] = ir_text.count("global")     / 100.0               # global vars
        vec[44] = ir_text.count("metadata")   / 100.0               # debug info density

        # Dims 45-255: zeros — reserved for GNN upgrade
        return vec


# ---------------------------------------------------------------------------
# Memory interface (stub — full version in src/memory/experience_store.py)
# ---------------------------------------------------------------------------

class MemoryStub:
    """
    Placeholder for pgvector experience store.
    Returns empty results until PostgreSQL is connected.
    Swap this out for ExperienceStore from memory/experience_store.py.
    """

    def query_similar(self, embedding: np.ndarray, top_k: int = 5) -> list:
        """
        In the real version: SELECT ... ORDER BY embedding <-> $1 LIMIT k
        For now: returns empty list so Boss Agent runs with zero memory.
        """
        return []

    def store(self, embedding: np.ndarray, plan: CompilationPlan,
              reward: float, metadata: dict):
        """Will write to PostgreSQL. No-op for now."""
        pass


# ---------------------------------------------------------------------------
# Boss Agent — core class
# ---------------------------------------------------------------------------

class BossAgent:
    """
    Orchestrates the AGentic_C compilation pipeline.

    Responsibilities:
      1. Encode incoming IR into a vector
      2. Query episodic memory for similar past compilations
      3. Make a routing decision (which agents, what budget)
      4. Return a CompilationPlan
      5. Store the outcome back to memory after compilation

    The Boss Agent does NOT use PPO in this iteration —
    its policy is rule-based + memory-informed.
    PPO will be layered on top once we have enough experience data.
    This is academically honest: you need data before you can train.
    """

    def __init__(self, config_path: str = "configs/config.yaml"):
        self.config = load_config(config_path)
        self.boss_cfg = self.config["agents"]["boss"]
        self.encoder = SimpleIREncoder()
        self.memory = MemoryStub()
        self._log("Boss Agent initialized.")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def decide(self, context: CompilationContext) -> CompilationPlan:
        """
        Main entry point. Takes a CompilationContext, returns a CompilationPlan.
        Called by pipeline.py before any agent fires.
        """
        self._log(f"Incoming job: {context.source_path} [{context.source_lang}]")

        # Step 1: encode IR if available
        if context.ir_embedding is None:
            ir_text = self._read_ir(context)
            if ir_text:
                context.ir_embedding = self.encoder.encode(ir_text)
                self._log(f"IR encoded → shape {context.ir_embedding.shape}")
            else:
                self._log("No IR yet — will run Pre-IR Fixer first.")

        # Step 2: query memory for similar past compilations
        past = []
        if context.ir_embedding is not None:
            past = self.memory.query_similar(
                context.ir_embedding,
                top_k=self.boss_cfg["top_k_memory"]
            )
            if past:
                self._log(f"Memory hit: {len(past)} similar past compilations found.")

        # Step 3: build the plan
        plan = self._build_plan(context, past)

        self._log(f"Plan decided → "
                  f"pre_fixer={plan.run_pre_fixer} | "
                  f"post_fixer={plan.run_post_fixer} | "
                  f"ir_tuner={plan.run_ir_tuner}(budget={plan.ir_tuner_budget}) | "
                  f"hw_tuner={plan.run_hw_tuner}(budget={plan.hw_tuner_budget})")

        return plan

    def store_outcome(self, context: CompilationContext,
                      plan: CompilationPlan, reward: float):
        """
        Called by pipeline.py after compilation completes.
        Stores the experience so future compilations can learn from it.
        """
        if context.ir_embedding is not None:
            self.memory.store(
                embedding=context.ir_embedding,
                plan=plan,
                reward=reward,
                metadata={
                    "source_lang": context.source_lang,
                    "target_arch": context.target_arch,
                    "source_path": context.source_path,
                }
            )
            self._log(f"Experience stored (reward={reward:.3f})")

    # ------------------------------------------------------------------
    # Internal — plan construction
    # ------------------------------------------------------------------

    def _build_plan(self, context: CompilationContext,
                    past_experiences: list) -> CompilationPlan:
        """
        Core routing logic.
        Order of priority:
          1. Memory-informed decision (if strong past match)
          2. Heuristic rules (language, arch, budget)
          3. Default: run everything
        """
        plan = CompilationPlan(
            hw_target=context.target_arch,
        )

        total_budget = context.optimization_budget

        # --- Memory-informed path ---
        if past_experiences:
            best_past = max(past_experiences, key=lambda x: x.get("reward", 0))
            if best_past.get("reward", 0) > 0.75:
                # Strong memory hit — mirror the past plan that worked well
                past_plan = best_past.get("plan", {})
                plan.ir_tuner_budget  = past_plan.get("ir_tuner_budget", 25)
                plan.hw_tuner_budget  = past_plan.get("hw_tuner_budget", 10)
                plan.based_on_memory  = True
                plan.confidence       = best_past["reward"]
                self._log("Memory-informed plan applied.")
                return plan

        # --- Heuristic rules ---

        # Budget allocation: split between IR and HW tuners
        # IR tuner gets 2/3 of budget, HW tuner gets 1/3
        plan.ir_tuner_budget = int(total_budget * 0.67)
        plan.hw_tuner_budget = int(total_budget * 0.33)

        # Language-specific adjustments
        if context.source_lang == "cpp":
            # C++ has more inlining opportunities
            plan.ir_tuner_budget += 5
            plan.pre_fixer_focus = "both"    # syntax + security

        elif context.source_lang == "c":
            plan.post_fixer_focus = "security"  # focus on memory safety

        # Architecture-specific adjustments
        if "arm64" in context.target_arch:
            plan.hw_target = "arm64-apple-macosx"
            plan.hw_tuner_budget = max(plan.hw_tuner_budget, 10)

        plan.confidence = 0.5   # medium confidence on heuristic path
        return plan

    # ------------------------------------------------------------------
    # Internal — helpers
    # ------------------------------------------------------------------

    def _read_ir(self, context: CompilationContext) -> Optional[str]:
        """Read IR file from disk if it exists."""
        ir_dir  = self.config["compiler"]["ir_output_dir"]
        stem    = Path(context.source_path).stem
        ir_path = os.path.join(ir_dir, f"{stem}.ll")

        if os.path.exists(ir_path):
            with open(ir_path, "r") as f:
                return f.read()
        return None

    def _log(self, msg: str):
        print(f"[BossAgent] {msg}")


# ---------------------------------------------------------------------------
# Smoke test — run directly to verify Boss Agent works
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import tempfile

    print("=" * 60)
    print("AGentic_C — Boss Agent Smoke Test")
    print("=" * 60)

    # Minimal config for standalone test
    test_config = {
        "compiler": {
            "frontend": "clang",
            "target_arch": "arm64-apple-macosx",
            "opt_level": "O0",
            "ir_output_dir": "/tmp/agentic_c/ir"
        },
        "agents": {
            "boss": {"top_k_memory": 5, "max_retries": 3},
            "fixer": {"max_repair_attempts": 3, "codebert_model": "microsoft/codebert-base"},
            "ir_tuner": {"max_steps": 45, "reward_metric": "IrInstructionCount"},
            "hw_tuner": {"max_steps": 30, "target": "llvm"}
        },
        "ppo": {"learning_rate": 0.0003, "n_steps": 2048, "batch_size": 64,
                "n_epochs": 10, "gamma": 0.99},
        "rewards": {"perf_weight": 0.5, "security_weight": 0.35, "size_weight": 0.15},
        "memory": {"host": "localhost", "port": 5432, "db": "agentic_c", "vector_dim": 256}
    }

    # Write temp config
    import yaml, tempfile
    tmp_cfg = tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False)
    yaml.dump(test_config, tmp_cfg)
    tmp_cfg.close()

    # Sample IR (what Clang produces from a simple C file)
    sample_ir = """
    define i32 @add(i32 %0, i32 %1) {
      %3 = alloca i32, align 4
      %4 = alloca i32, align 4
      store i32 %0, ptr %3, align 4
      store i32 %1, ptr %4, align 4
      %5 = load i32, ptr %3, align 4
      %6 = load i32, ptr %4, align 4
      %7 = add nsw i32 %5, %6
      ret i32 %7
    }
    define i32 @main() {
      %1 = alloca i32, align 4
      store i32 0, ptr %1, align 4
      %2 = call i32 @add(i32 1, i32 2)
      ret i32 %2
    }
    """

    # Encode IR
    encoder = SimpleIREncoder()
    embedding = encoder.encode(sample_ir)
    print(f"\n✓ IR encoded: shape={embedding.shape}, "
          f"non-zero dims={np.count_nonzero(embedding)}")

    # Build context
    ctx = CompilationContext(
        source_path="/tmp/test.c",
        source_lang="c",
        target_arch="arm64-apple-macosx",
        ir_embedding=embedding,
        optimization_budget=45
    )

    # Run Boss Agent (using temp config)
    agent = BossAgent(config_path=tmp_cfg.name)
    plan  = agent.decide(ctx)

    print(f"\n✓ Plan generated:")
    print(f"  pre_fixer     : {plan.run_pre_fixer} (focus={plan.pre_fixer_focus})")
    print(f"  post_fixer    : {plan.run_post_fixer} (focus={plan.post_fixer_focus})")
    print(f"  ir_tuner      : {plan.run_ir_tuner} (budget={plan.ir_tuner_budget})")
    print(f"  hw_tuner      : {plan.run_hw_tuner} (budget={plan.hw_tuner_budget})")
    print(f"  confidence    : {plan.confidence}")
    print(f"  memory-based  : {plan.based_on_memory}")

    # Simulate storing outcome
    agent.store_outcome(ctx, plan, reward=0.82)

    print("\n✓ Boss Agent smoke test PASSED")
    print("=" * 60)