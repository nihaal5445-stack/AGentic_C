"""
AGentic_C — Frontend (src/core/frontend.py)
Clang wrapper. Converts source code to LLVM IR.
This is the entry gate — nothing downstream runs without this succeeding.
"""

import os
import subprocess
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional
import yaml


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class FrontendResult:
    """
    Everything the pipeline needs to know about a compilation attempt.
    Returned by Frontend.compile() regardless of success or failure.
    """
    success: bool
    source_path: str
    ir_path: Optional[str] = None          # set on success

    # Error info — set on failure, used by Pre-IR Fixer
    error_type: Optional[str] = None       # 'syntax' | 'semantic' | 'system'
    error_message: Optional[str] = None    # raw clang stderr
    error_lines: list = field(default_factory=list)  # parsed [(line, col, msg)]

    # Metadata
    clang_version: Optional[str] = None
    target_triple: Optional[str] = None
    compile_time_ms: float = 0.0


@dataclass
class ParsedError:
    """A single structured error extracted from Clang's stderr output."""
    filepath: str
    line: int
    col: int
    severity: str      # 'error' | 'warning' | 'note'
    message: str
    raw: str           # original line from stderr


# ---------------------------------------------------------------------------
# Config loader
# ---------------------------------------------------------------------------

def load_config(config_path: str = "configs/config.yaml") -> dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


# ---------------------------------------------------------------------------
# Frontend class
# ---------------------------------------------------------------------------

class Frontend:
    """
    Wraps Clang to convert source code into LLVM IR.

    Responsibilities:
      - Locate clang binary
      - Run clang -O0 -emit-llvm on source files
      - Parse structured errors from stderr
      - Return FrontendResult (success or failure with full error info)

    Does NOT fix errors — that's the Pre-IR Fixer's job.
    Does NOT optimize IR — that's the IR Tuner's job.
    Just compiles cleanly and reports what happened.
    """

    # Source languages we support and their file extensions
    SUPPORTED_EXTENSIONS = {
        ".c":   "c",
        ".cpp": "cpp",
        ".cc":  "cpp",
        ".cxx": "cpp",
    }

    def __init__(self, config_path: str = "configs/config.yaml"):
        self.config      = load_config(config_path)
        self.compiler_cfg = self.config["compiler"]
        self.ir_output_dir = self.compiler_cfg["ir_output_dir"]
        self.target_arch   = self.compiler_cfg["target_arch"]
        self.clang_path    = self._find_clang()

        # Ensure IR output directory exists
        os.makedirs(self.ir_output_dir, exist_ok=True)

        self._log(f"Frontend initialized.")
        self._log(f"Clang: {self.clang_path}")
        self._log(f"IR output dir: {self.ir_output_dir}")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def compile(self, source_path: str) -> FrontendResult:
        """
        Main entry point. Compile source → LLVM IR.

        Args:
            source_path: absolute or relative path to source file

        Returns:
            FrontendResult with success=True and ir_path set,
            OR success=False with error info for Pre-IR Fixer.
        """
        import time

        source_path = os.path.abspath(source_path)
        self._log(f"Compiling: {source_path}")

        # Step 1: validate source file
        validation_error = self._validate_source(source_path)
        if validation_error:
            return validation_error

        # Step 2: determine output IR path
        ir_path = self._get_ir_path(source_path)

        # Step 3: build clang command
        cmd = self._build_command(source_path, ir_path)
        self._log(f"Command: {' '.join(cmd)}")

        # Step 4: run clang
        start = time.time()
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=30       # 30s timeout — compilation shouldn't take longer
            )
            elapsed_ms = (time.time() - start) * 1000

        except subprocess.TimeoutExpired:
            return FrontendResult(
                success=False,
                source_path=source_path,
                error_type="system",
                error_message="Clang timed out after 30 seconds.",
                error_lines=[]
            )
        except FileNotFoundError:
            return FrontendResult(
                success=False,
                source_path=source_path,
                error_type="system",
                error_message=f"Clang not found at: {self.clang_path}",
                error_lines=[]
            )

        # Step 5: check result
        if result.returncode == 0 and os.path.exists(ir_path):
            self._log(f"SUCCESS → {ir_path} ({elapsed_ms:.1f}ms)")
            return FrontendResult(
                success=True,
                source_path=source_path,
                ir_path=ir_path,
                clang_version=self._get_clang_version(),
                target_triple=self.target_arch,
                compile_time_ms=elapsed_ms
            )

        else:
            # Parse clang's stderr into structured errors
            stderr = result.stderr + result.stdout  # clang mixes these
            parsed_errors = self._parse_clang_errors(stderr)
            error_type = self._classify_error(parsed_errors, stderr)

            self._log(f"FAILED ({error_type}) — {len(parsed_errors)} errors found")
            for e in parsed_errors[:3]:  # log first 3
                self._log(f"  line {e.line}: {e.message}")

            return FrontendResult(
                success=False,
                source_path=source_path,
                error_type=error_type,
                error_message=stderr,
                error_lines=parsed_errors,
                compile_time_ms=elapsed_ms
            )

    def get_ir_text(self, ir_path: str) -> Optional[str]:
        """Read IR file contents. Used by Boss Agent encoder."""
        if ir_path and os.path.exists(ir_path):
            with open(ir_path, "r") as f:
                return f.read()
        return None

    def detect_language(self, source_path: str) -> str:
        """Detect source language from file extension."""
        ext = Path(source_path).suffix.lower()
        return self.SUPPORTED_EXTENSIONS.get(ext, "unknown")

    # ------------------------------------------------------------------
    # Internal — command building
    # ------------------------------------------------------------------

    def _build_command(self, source_path: str, ir_path: str) -> list:
        """
        Build the clang command list.

        -O0              → no optimization, raw IR (agents do the optimizing)
        -emit-llvm       → output LLVM IR instead of object code
        -S               → output text format (.ll) not binary (.bc)
        -Wno-everything  → suppress warnings, we only care about errors
        -ferror-limit=20 → cap errors at 20 (avoid megabyte stderr dumps)
        """
        cmd = [
            self.clang_path,
            "-O0",
            "-emit-llvm",
            "-S",
            "-Wno-everything",       # suppress warnings during IR generation
            "-ferror-limit=20",      # cap error output
            source_path,
            "-o", ir_path,
        ]

        # Add target triple only if not using system default
        # (system default is usually correct for native compilation)
        # Uncomment if cross-compilation is needed:
        # cmd += [f"--target={self.target_arch}"]

        return cmd

    # ------------------------------------------------------------------
    # Internal — validation
    # ------------------------------------------------------------------

    def _validate_source(self, source_path: str) -> Optional[FrontendResult]:
        """Return an error FrontendResult if source file is invalid, else None."""

        if not os.path.exists(source_path):
            return FrontendResult(
                success=False,
                source_path=source_path,
                error_type="system",
                error_message=f"Source file not found: {source_path}"
            )

        if not os.path.isfile(source_path):
            return FrontendResult(
                success=False,
                source_path=source_path,
                error_type="system",
                error_message=f"Path is not a file: {source_path}"
            )

        ext = Path(source_path).suffix.lower()
        if ext not in self.SUPPORTED_EXTENSIONS:
            return FrontendResult(
                success=False,
                source_path=source_path,
                error_type="system",
                error_message=f"Unsupported extension '{ext}'. "
                              f"Supported: {list(self.SUPPORTED_EXTENSIONS.keys())}"
            )

        if os.path.getsize(source_path) == 0:
            return FrontendResult(
                success=False,
                source_path=source_path,
                error_type="syntax",
                error_message="Source file is empty."
            )

        return None  # all good

    # ------------------------------------------------------------------
    # Internal — error parsing
    # ------------------------------------------------------------------

    def _parse_clang_errors(self, stderr: str) -> list:
        """
        Parse Clang's stderr into structured ParsedError objects.

        Clang error format:
          /path/to/file.c:LINE:COL: SEVERITY: MESSAGE
          e.g.:
          /tmp/test.c:3:5: error: use of undeclared identifier 'x'
        """
        errors = []
        for line in stderr.split("\n"):
            line = line.strip()
            if not line:
                continue

            # Try to parse structured clang error line
            # Format: filepath:line:col: severity: message
            parts = line.split(":")
            if len(parts) >= 5:
                try:
                    filepath = parts[0]
                    lineno   = int(parts[1])
                    colno    = int(parts[2])
                    severity = parts[3].strip()
                    message  = ":".join(parts[4:]).strip()

                    if severity in ("error", "warning", "note", "fatal error"):
                        errors.append(ParsedError(
                            filepath=filepath,
                            line=lineno,
                            col=colno,
                            severity=severity,
                            message=message,
                            raw=line
                        ))
                except (ValueError, IndexError):
                    pass  # line didn't match expected format, skip

        return errors

    def _classify_error(self, parsed_errors: list, raw_stderr: str) -> str:
        """
        Classify the type of compilation failure.
        Pre-IR Fixer uses this to decide what kind of repair to attempt.

        Returns: 'syntax' | 'semantic' | 'system'
        """
        if not parsed_errors:
            # No structured errors parsed — likely a system/linking issue
            return "system"

        # Look at error messages to distinguish syntax vs semantic
        syntax_keywords = [
            "expected", "unexpected", "unterminated",
            "missing", "stray", "invalid token",
            "expected ';'", "expected ')'", "expected '}'",
        ]
        semantic_keywords = [
            "undeclared", "undefined", "no member",
            "incompatible", "cannot convert", "redefinition",
            "conflicting types", "too many arguments",
            "use of undeclared identifier"
        ]

        error_messages = " ".join(
            e.message.lower() for e in parsed_errors if e.severity == "error"
        )

        syntax_score   = sum(1 for kw in syntax_keywords if kw in error_messages)
        semantic_score = sum(1 for kw in semantic_keywords if kw in error_messages)

        if syntax_score > semantic_score:
            return "syntax"
        elif semantic_score > 0:
            return "semantic"
        else:
            return "syntax"   # default to syntax — more commonly fixable

    # ------------------------------------------------------------------
    # Internal — helpers
    # ------------------------------------------------------------------

    def _get_ir_path(self, source_path: str) -> str:
        """Compute output IR file path from source path."""
        stem = Path(source_path).stem    # 'test' from '/tmp/test.c'
        return os.path.join(self.ir_output_dir, f"{stem}.ll")

    def _find_clang(self) -> str:
        """
        Find clang binary. Priority:
          1. 'clang' on system PATH (Xcode clang on Mac)
          2. Common locations
        """
        # Try system PATH first
        clang = shutil.which("clang")
        if clang:
            return clang

        # Common Mac locations
        candidates = [
            "/usr/bin/clang",
            "/Applications/Xcode.app/Contents/Developer/Toolchains/"
            "XcodeDefault.xctoolchain/usr/bin/clang",
        ]
        for c in candidates:
            if os.path.exists(c):
                return c

        raise RuntimeError(
            "Clang not found. Install Xcode Command Line Tools:\n"
            "  xcode-select --install"
        )

    def _get_clang_version(self) -> str:
        """Get clang version string for metadata."""
        try:
            result = subprocess.run(
                [self.clang_path, "--version"],
                capture_output=True, text=True, timeout=5
            )
            return result.stdout.split("\n")[0]
        except Exception:
            return "unknown"

    def _log(self, msg: str):
        print(f"[Frontend] {msg}")


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import tempfile

    print("=" * 60)
    print("AGentic_C — Frontend Smoke Test")
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
            "fixer": {"max_repair_attempts": 3,
                      "codebert_model": "microsoft/codebert-base"},
            "ir_tuner": {"max_steps": 45,
                         "reward_metric": "IrInstructionCount"},
            "hw_tuner": {"max_steps": 30, "target": "llvm"}
        },
        "ppo": {"learning_rate": 0.0003, "n_steps": 2048,
                "batch_size": 64, "n_epochs": 10, "gamma": 0.99},
        "rewards": {"perf_weight": 0.5,
                    "security_weight": 0.35, "size_weight": 0.15},
        "memory": {"host": "localhost", "port": 5432,
                   "db": "agentic_c", "vector_dim": 256}
    }

    import yaml
    tmp_cfg = tempfile.NamedTemporaryFile(
        mode="w", suffix=".yaml", delete=False)
    yaml.dump(test_config, tmp_cfg)
    tmp_cfg.close()

    frontend = Frontend(config_path=tmp_cfg.name)

    # ── Test 1: Valid C file ──────────────────────────────────────────
    print("\n── Test 1: Valid C file ──")
    valid_src = tempfile.NamedTemporaryFile(
        mode="w", suffix=".c", delete=False)
    valid_src.write("""
#include <stdio.h>
int add(int a, int b) { return a + b; }
int main() {
    int result = add(3, 4);
    printf("Result: %d\\n", result);
    return 0;
}
""")
    valid_src.close()

    result = frontend.compile(valid_src.name)
    print(f"  success      : {result.success}")
    print(f"  ir_path      : {result.ir_path}")
    print(f"  compile_time : {result.compile_time_ms:.1f}ms")
    print(f"  clang        : {result.clang_version}")
    assert result.success, "Test 1 FAILED — valid C should compile"
    print("  ✓ PASSED")

    # Read and show a snippet of the IR
    ir_text = frontend.get_ir_text(result.ir_path)
    ir_lines = [l for l in ir_text.split("\n") if l.strip()][:6]
    print(f"\n  IR snippet (first 6 non-empty lines):")
    for line in ir_lines:
        print(f"    {line}")

    # ── Test 2: Syntax Error ──────────────────────────────────────────
    print("\n── Test 2: Broken C file (syntax error) ──")
    broken_src = tempfile.NamedTemporaryFile(
        mode="w", suffix=".c", delete=False)
    broken_src.write("""
int main() {
    int x = 
    return x
}
""")
    broken_src.close()

    result2 = frontend.compile(broken_src.name)
    print(f"  success      : {result2.success}")
    print(f"  error_type   : {result2.error_type}")
    print(f"  errors found : {len(result2.error_lines)}")
    for err in result2.error_lines[:3]:
        print(f"    line {err.line}: [{err.severity}] {err.message}")
    assert not result2.success, "Test 2 FAILED — broken C should fail"
    print("  ✓ PASSED — error correctly captured")

    # ── Test 3: Nonexistent file ──────────────────────────────────────
    print("\n── Test 3: Nonexistent file ──")
    result3 = frontend.compile("/tmp/does_not_exist.c")
    print(f"  success     : {result3.success}")
    print(f"  error_type  : {result3.error_type}")
    print(f"  message     : {result3.error_message}")
    assert not result3.success
    print("  ✓ PASSED")

    # ── Test 4: Language detection ────────────────────────────────────
    print("\n── Test 4: Language detection ──")
    cases = [
        ("/tmp/foo.c",   "c"),
        ("/tmp/bar.cpp", "cpp"),
        ("/tmp/baz.cc",  "cpp"),
        ("/tmp/x.py",    "unknown"),
    ]
    for path, expected in cases:
        detected = frontend.detect_language(path)
        status = "✓" if detected == expected else "✗"
        print(f"  {status} {path} → {detected}")

    print("\n" + "=" * 60)
    print("✓ Frontend smoke test PASSED")
    print("=" * 60)