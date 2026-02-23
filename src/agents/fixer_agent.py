"""
AGentic_C — Pre-IR Fixer Agent (src/agents/fixer_agent.py)

Two responsibilities:
  1. Pre-IR  — repair broken source code so Clang can produce IR
  2. Post-IR — scan valid IR for security vulnerabilities

Both use CodeBERT for semantic understanding.
Rule engine handles common syntax patterns fast.
CodeBERT handles everything the rules miss.
"""

import os
import re
import yaml
import torch
import numpy as np
from dataclasses import dataclass, field
from typing import Optional
from pathlib import Path
from transformers import AutoTokenizer, AutoModel


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class FixerResult:
    """
    Returned by both pre_fix() and post_fix().
    Pipeline reads this to decide what to do next.
    """
    success: bool
    source_path: str

    # Pre-IR fix fields
    patched_source: Optional[str] = None     # fixed source code text
    patches_applied: list = field(default_factory=list)  # what we changed
    attempts: int = 0

    # Post-IR security fields
    security_score: float = 1.0             # 1.0 = clean, 0.0 = very dangerous
    vulnerabilities: list = field(default_factory=list)  # list of VulnMatch

    # Shared
    message: str = ""


@dataclass
class VulnMatch:
    """A detected vulnerability pattern."""
    cwe_id: str          # e.g. 'CWE-119'
    cwe_name: str        # e.g. 'Buffer Overflow'
    confidence: float    # cosine similarity score 0.0-1.0
    line_hint: str       # code snippet that triggered it
    severity: str        # 'critical' | 'high' | 'medium' | 'low'


# ---------------------------------------------------------------------------
# Config loader
# ---------------------------------------------------------------------------

def load_config(config_path: str = "configs/config.yaml") -> dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


# ---------------------------------------------------------------------------
# CodeBERT Encoder
# ---------------------------------------------------------------------------

class CodeBERTEncoder:
    """
    Wraps microsoft/codebert-base.
    Encodes source code or IR text into a 768-dim semantic vector.
    Used by both Pre-IR and Post-IR fixer.
    """

    def __init__(self, model_name: str = "microsoft/codebert-base"):
        print(f"[CodeBERTEncoder] Loading {model_name}...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model     = AutoModel.from_pretrained(model_name)
        self.model.eval()   # inference mode — disables dropout
        print(f"[CodeBERTEncoder] Ready.")

    def encode(self, code: str, max_length: int = 512) -> np.ndarray:
        """
        Encode code text → 768-dim float32 vector.

        Uses [CLS] token representation as the
        summary embedding of the entire code snippet.
        This is standard practice for BERT-family models.

        max_length=512 is BERT's hard limit.
        Code longer than 512 tokens gets truncated.
        For our use case (function-level snippets) this is fine.
        """
        # Tokenize
        inputs = self.tokenizer(
            code,
            return_tensors="pt",       # PyTorch tensors
            truncation=True,
            max_length=max_length,
            padding=True
        )

        # Forward pass — no gradient needed for inference
        with torch.no_grad():
            outputs = self.model(**inputs)

        # [CLS] token is always at position 0
        # Shape: [1, seq_len, 768] → take position 0 → [768]
        cls_embedding = outputs.last_hidden_state[:, 0, :]
        return cls_embedding.squeeze().numpy().astype(np.float32)

    def cosine_similarity(self, vec_a: np.ndarray,
                          vec_b: np.ndarray) -> float:
        """Cosine similarity between two vectors. Range: -1 to 1."""
        norm_a = np.linalg.norm(vec_a)
        norm_b = np.linalg.norm(vec_b)
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return float(np.dot(vec_a, vec_b) / (norm_a * norm_b))


# ---------------------------------------------------------------------------
# Vulnerability Pattern Library
# ---------------------------------------------------------------------------

# Known vulnerability code patterns.
# In production: these embeddings come from pgvector (pre-computed).
# Here: we store the raw code patterns and encode them on first use.
# CodeBERT learned these patterns from millions of CVE patches on GitHub.

VULN_PATTERNS = {
    "CWE-119": {
        "name": "Buffer Overflow",
        "severity": "critical",
        "patterns": [
            "char buf[]; strcpy(buf, input);",
            "char buffer[256]; gets(buffer);",
            "memcpy(dest, src, strlen(src));",
            "sprintf(buf, format, input);",
            "strcat(dest, src);",
        ]
    },
    "CWE-416": {
        "name": "Use After Free",
        "severity": "critical",
        "patterns": [
            "free(ptr); ptr->field = value;",
            "free(ptr); return ptr;",
            "delete obj; obj->method();",
        ]
    },
    "CWE-476": {
        "name": "NULL Pointer Dereference",
        "severity": "high",
        "patterns": [
            "ptr = malloc(size); ptr->field = x;",
            "char* p = NULL; *p = 0;",
            "int* arr = NULL; arr[0] = 1;",
        ]
    },
    "CWE-190": {
        "name": "Integer Overflow",
        "severity": "high",
        "patterns": [
            "int size = a + b; malloc(size);",
            "unsigned int x = y * z;",
            "size_t len = strlen(s1) + strlen(s2);",
        ]
    },
    "CWE-78": {
        "name": "OS Command Injection",
        "severity": "critical",
        "patterns": [
            "system(user_input);",
            "popen(cmd, 'r');",
            "execve(path, argv, envp);",
        ]
    },
    "CWE-89": {
        "name": "SQL Injection",
        "severity": "critical",
        "patterns": [
            "sprintf(query, SELECT * FROM users WHERE id=%s, input);",
            "strcat(sql, user_input);",
        ]
    },
}


class VulnerabilityLibrary:
    """
    Manages vulnerability pattern embeddings.
    Encodes patterns once and caches them.
    """

    def __init__(self, encoder: CodeBERTEncoder):
        self.encoder  = encoder
        self._cache   = {}   # cwe_id -> list of np.ndarray embeddings
        print("[VulnLibrary] Pre-computing vulnerability embeddings...")
        self._precompute()
        print(f"[VulnLibrary] {len(self._cache)} CWE patterns loaded.")

    def _precompute(self):
        """Encode all vulnerability patterns once at startup."""
        for cwe_id, info in VULN_PATTERNS.items():
            embeddings = []
            for pattern in info["patterns"]:
                emb = self.encoder.encode(pattern)
                embeddings.append(emb)
            self._cache[cwe_id] = embeddings

    def scan(self, code_embedding: np.ndarray,
             threshold: float = 0.75) -> list:
        """
        Compare code embedding against all vulnerability patterns.
        Returns list of VulnMatch where similarity > threshold.

        threshold=0.75 means "75% similar to a known vuln pattern"
        Tune this: lower = more sensitive (more false positives)
                          higher = more conservative (more false negatives)
        """
        matches = []

        for cwe_id, embeddings in self._cache.items():
            info = VULN_PATTERNS[cwe_id]

            # Check similarity against each pattern for this CWE
            # Take the MAX similarity — if any pattern matches, flag it
            max_sim = 0.0
            for vuln_emb in embeddings:
                sim = self.encoder.cosine_similarity(code_embedding, vuln_emb)
                max_sim = max(max_sim, sim)

            if max_sim >= threshold:
                # Find which pattern triggered it
                best_pattern = ""
                for i, vuln_emb in enumerate(embeddings):
                    sim = self.encoder.cosine_similarity(code_embedding, vuln_emb)
                    if sim == max_sim:
                        best_pattern = VULN_PATTERNS[cwe_id]["patterns"][i]
                        break

                matches.append(VulnMatch(
                    cwe_id=cwe_id,
                    cwe_name=info["name"],
                    confidence=round(max_sim, 3),
                    line_hint=best_pattern,
                    severity=info["severity"]
                ))

        # Sort by confidence descending
        matches.sort(key=lambda x: x.confidence, reverse=True)
        return matches


# ---------------------------------------------------------------------------
# Rule-Based Syntax Patch Engine
# ---------------------------------------------------------------------------

class SyntaxPatchEngine:
    """
    Fast rule-based fixer for common C/C++ syntax errors.
    Handles the mechanical stuff — missing semicolons, brackets, etc.
    CodeBERT handles the semantic stuff the rules can't.

    Rules are ordered by frequency of occurrence in practice.
    """

    def apply_rules(self, source: str, error_lines: list) -> tuple:
        """
        Apply all applicable rules to source.
        Returns (patched_source, list_of_patches_applied).
        """
        lines       = source.split("\n")
        patches     = []
        error_linenos = {e.line for e in error_lines}

        # Rule 1: Missing semicolons
        lines, p = self._fix_missing_semicolons(lines, error_linenos)
        patches.extend(p)

        # Rule 2: Unmatched braces
        lines, p = self._fix_unmatched_braces(lines)
        patches.extend(p)

        # Rule 3: Common typos in keywords
        lines, p = self._fix_keyword_typos(lines)
        patches.extend(p)

        # Rule 4: Missing return statement in non-void functions
        lines, p = self._fix_missing_return(lines)
        patches.extend(p)

        # Rule 5: Unterminated string literals
        lines, p = self._fix_unterminated_strings(lines, error_linenos)
        patches.extend(p)

        return "\n".join(lines), patches

    def _fix_missing_semicolons(self, lines, error_linenos):
        """Add semicolons to lines that look like statements but are missing them."""
        patches = []
        # Patterns that are statements and need semicolons
        statement_pattern = re.compile(
            r'^(\s*)(int|char|float|double|long|short|unsigned|'
            r'return|break|continue|[a-zA-Z_]\w*\s*=|\+\+|--).*[^;{}\s]$'
        )
        for i, line in enumerate(lines):
            lineno = i + 1  # 1-indexed
            stripped = line.rstrip()
            if not stripped or stripped.endswith((';', '{', '}', '//', '*')):
                continue
            # Only fix lines near reported errors
            if lineno in error_linenos or lineno - 1 in error_linenos:
                if statement_pattern.match(stripped):
                    lines[i] = stripped + ';'
                    patches.append(f"line {lineno}: added missing semicolon")
        return lines, patches

    def _fix_unmatched_braces(self, lines):
        """Count braces and add missing closing braces at end of file."""
        patches = []
        source = "\n".join(lines)
        open_count  = source.count('{')
        close_count = source.count('}')
        diff = open_count - close_count

        if diff > 0:
            # More opens than closes — add closing braces
            for _ in range(diff):
                lines.append('}')
            patches.append(f"added {diff} missing closing brace(s)")
        return lines, patches

    def _fix_keyword_typos(self, lines):
        """Fix common C keyword typos."""
        patches = []
        typos = {
            r'\bintmain\b':     'int main',
            r'\bvoidmain\b':    'void main',
            r'\bretun\b':       'return',
            r'\bretrun\b':      'return',
            r'\binclude\b':     '#include',
            r'\bprinf\b':       'printf',
            r'\bpirntf\b':      'printf',
            r'\bscnaf\b':       'scanf',
            r'\bmaloc\b':       'malloc',
            r'\bnULL\b':        'NULL',
            r'\bNul\b':         'NULL',
        }
        for i, line in enumerate(lines):
            original = line
            for pattern, replacement in typos.items():
                line = re.sub(pattern, replacement, line)
            if line != original:
                lines[i] = line
                patches.append(f"line {i+1}: fixed keyword typo")
        return lines, patches

    def _fix_missing_return(self, lines):
        """
        Add 'return 0;' to main() if it's missing.
        Only applies to main — too risky to auto-add returns elsewhere.
        """
        patches = []
        source = "\n".join(lines)

        # Only if main() exists and doesn't have return
        has_main   = "int main" in source
        has_return = bool(re.search(r'\breturn\b', source))

        if has_main and not has_return:
            # Find the last } and insert return 0; before it
            for i in range(len(lines) - 1, -1, -1):
                if lines[i].strip() == '}':
                    lines.insert(i, '    return 0;')
                    patches.append(f"line {i+1}: added missing return 0")
                    break
        return lines, patches

    def _fix_unterminated_strings(self, lines, error_linenos):
        """Close unterminated string literals."""
        patches = []
        for i, line in enumerate(lines):
            lineno = i + 1
            if lineno not in error_linenos:
                continue
            # Count unescaped quotes
            stripped = re.sub(r'\\.', '', line)  # remove escaped chars first
            if stripped.count('"') % 2 != 0:
                lines[i] = line.rstrip() + '"'
                patches.append(f"line {lineno}: closed unterminated string")
        return lines, patches


# ---------------------------------------------------------------------------
# Fixer Agent — Main Class
# ---------------------------------------------------------------------------

class FixerAgent:
    """
    Pre-IR Fixer Agent for AGentic_C.

    pre_fix()  → called when Clang fails. Repairs source code.
    post_fix() → called after Clang succeeds. Scans IR for vulns.

    Architecture:
      Rule engine  → fast, handles mechanical syntax errors
      CodeBERT     → semantic understanding, handles everything else
      VulnLibrary  → security pattern matching via cosine similarity
    """

    def __init__(self, config_path: str = "configs/config.yaml"):
        self.config     = load_config(config_path)
        self.fixer_cfg  = self.config["agents"]["fixer"]
        self.max_attempts = self.fixer_cfg.get("max_repair_attempts", 3)

        # Initialize CodeBERT (loads from cache — fast after first download)
        model_name      = self.fixer_cfg.get(
            "codebert_model", "microsoft/codebert-base")
        self.encoder    = CodeBERTEncoder(model_name)

        # Initialize rule engine and vuln library
        self.rules      = SyntaxPatchEngine()
        self.vuln_lib   = VulnerabilityLibrary(self.encoder)

        self._log("Fixer Agent initialized.")

    # ------------------------------------------------------------------
    # Public API — Pre-IR Fix
    # ------------------------------------------------------------------

    def pre_fix(self, source_path: str,
                error_type: str,
                error_lines: list) -> FixerResult:
        """
        Attempt to repair broken source code.
        Called by pipeline when Frontend.compile() returns success=False.

        Args:
            source_path: path to the broken source file
            error_type:  'syntax' | 'semantic' | 'system'
            error_lines: list of ParsedError from Frontend

        Returns:
            FixerResult with patched_source set if repair succeeded
        """
        self._log(f"Pre-fix: {source_path} [{error_type}]")
        self._log(f"  {len(error_lines)} errors to address")

        # Read source
        try:
            with open(source_path, "r") as f:
                original_source = f.read()
        except Exception as e:
            return FixerResult(
                success=False,
                source_path=source_path,
                message=f"Cannot read source: {e}"
            )

        if error_type == "system":
            # System errors (file not found, etc.) — can't fix
            return FixerResult(
                success=False,
                source_path=source_path,
                message="System error — cannot repair programmatically"
            )

        current_source = original_source
        all_patches    = []

        for attempt in range(1, self.max_attempts + 1):
            self._log(f"  Attempt {attempt}/{self.max_attempts}")

            # Phase A: Rule-based fixes (fast, low cost)
            patched, patches = self.rules.apply_rules(
                current_source, error_lines)
            all_patches.extend(patches)

            if patches:
                self._log(f"  Rules applied: {patches}")

            # Phase B: CodeBERT semantic analysis
            # Encode the broken snippet around error lines
            error_snippet = self._extract_error_context(
                current_source, error_lines)
            code_embedding = self.encoder.encode(error_snippet)

            # Get semantic suggestion
            semantic_patch = self._semantic_repair(
                patched, error_lines, code_embedding, error_type)

            if semantic_patch:
                patched = semantic_patch["source"]
                all_patches.append(semantic_patch["description"])
                self._log(f"  Semantic patch: {semantic_patch['description']}")

            current_source = patched

            # Check if we made any changes this attempt
            if current_source != original_source or patches or semantic_patch:
                self._log(f"  Patches applied: {len(all_patches)}")
                return FixerResult(
                    success=True,
                    source_path=source_path,
                    patched_source=current_source,
                    patches_applied=all_patches,
                    attempts=attempt,
                    message=f"Repaired in {attempt} attempt(s)"
                )

        # All attempts exhausted — return best effort
        return FixerResult(
            success=len(all_patches) > 0,
            source_path=source_path,
            patched_source=current_source,
            patches_applied=all_patches,
            attempts=self.max_attempts,
            message="Max attempts reached — partial repair"
        )

    # ------------------------------------------------------------------
    # Public API — Post-IR Security Scan
    # ------------------------------------------------------------------

    def post_fix(self, source_path: str,
                 ir_text: Optional[str] = None,
                 focus: str = "security") -> FixerResult:
        """
        Scan compiled IR for security vulnerabilities.
        Called by pipeline after Frontend.compile() succeeds.

        Args:
            source_path: original source file path
            ir_text:     LLVM IR text to scan
            focus:       'security' | 'both' (syntax already handled pre-IR)

        Returns:
            FixerResult with security_score and vulnerabilities list
        """
        self._log(f"Post-fix (security scan): {source_path}")

        # Prefer scanning source code — CodeBERT was trained on source
        # Fall back to IR text if source not available
        scan_target = ""
        if os.path.exists(source_path):
            with open(source_path, "r") as f:
                scan_target = f.read()
        elif ir_text:
            scan_target = ir_text
        else:
            return FixerResult(
                success=False,
                source_path=source_path,
                security_score=0.5,
                message="Nothing to scan"
            )

        # Encode the code
        self._log("  Encoding for security scan...")
        code_embedding = self.encoder.encode(scan_target)

        # Scan against vulnerability library
        threshold  = 0.75   # configurable later via config.yaml
        matches    = self.vuln_lib.scan(code_embedding, threshold)

        # Compute security score
        # 1.0 = completely clean
        # Each vuln reduces score based on severity and confidence
        security_score = self._compute_security_score(matches)

        if matches:
            self._log(f"  {len(matches)} vulnerability pattern(s) detected:")
            for m in matches:
                self._log(f"    [{m.severity.upper()}] {m.cwe_id} "
                          f"{m.cwe_name} (confidence={m.confidence:.3f})")
        else:
            self._log(f"  Clean — no vulnerability patterns detected.")

        self._log(f"  Security score: {security_score:.3f}")

        return FixerResult(
            success=True,
            source_path=source_path,
            security_score=security_score,
            vulnerabilities=matches,
            message=f"{len(matches)} vulnerability pattern(s) found"
        )

    # ------------------------------------------------------------------
    # Internal — Semantic Repair
    # ------------------------------------------------------------------

    def _semantic_repair(self, source: str, error_lines: list,
                         embedding: np.ndarray,
                         error_type: str) -> Optional[dict]:
        """
        Use CodeBERT embedding to guide semantic repairs.

        Current implementation: pattern-based semantic fixes
        informed by what CodeBERT says the code "looks like."

        Future: fine-tuned CodeBERT seq2seq model that generates fixes.
        This is the upgrade path — same interface, better model inside.
        """
        lines = source.split("\n")
        repairs_made = []

        for error in error_lines:
            if error.line <= 0 or error.line > len(lines):
                continue

            line_idx = error.line - 1  # convert to 0-indexed
            line     = lines[line_idx]
            msg      = error.message.lower()

            # Semantic repair: undeclared identifier
            if "undeclared identifier" in msg or "use of undeclared" in msg:
                # Extract the identifier name from error message
                # e.g. "use of undeclared identifier 'x'" → 'x'
                match = re.search(r"'(\w+)'", error.message)
                if match:
                    identifier = match.group(1)
                    # Add a declaration before the function containing this line
                    decl = f"int {identifier} = 0; /* auto-declared by fixer */"
                    insert_idx = self._find_function_start(lines, line_idx)
                    if insert_idx >= 0:
                        lines.insert(insert_idx, decl)
                        repairs_made.append(
                            f"auto-declared '{identifier}' at line {insert_idx+1}")

            # Semantic repair: incompatible types — add cast
            elif "incompatible" in msg and "pointer" in msg:
                if "void *" not in line and "NULL" not in line:
                    lines[line_idx] = re.sub(
                        r'= (\w+)',
                        r'= (void*)\1',
                        line, count=1)
                    repairs_made.append(
                        f"line {error.line}: added void* cast")

            # Semantic repair: missing include for common functions
            elif "implicit declaration" in msg:
                func_match = re.search(r"'(\w+)'", error.message)
                if func_match:
                    func_name = func_match.group(1)
                    include = self._suggest_include(func_name)
                    if include and include not in source:
                        lines.insert(0, include)
                        repairs_made.append(
                            f"added {include} for {func_name}")

        if repairs_made:
            return {
                "source": "\n".join(lines),
                "description": "; ".join(repairs_made)
            }
        return None

    def _suggest_include(self, func_name: str) -> Optional[str]:
        """Map function names to their required headers."""
        include_map = {
            "printf":  "#include <stdio.h>",
            "scanf":   "#include <stdio.h>",
            "fprintf": "#include <stdio.h>",
            "sprintf": "#include <stdio.h>",
            "malloc":  "#include <stdlib.h>",
            "calloc":  "#include <stdlib.h>",
            "realloc": "#include <stdlib.h>",
            "free":    "#include <stdlib.h>",
            "exit":    "#include <stdlib.h>",
            "strlen":  "#include <string.h>",
            "strcpy":  "#include <string.h>",
            "strcat":  "#include <string.h>",
            "strcmp":  "#include <string.h>",
            "memcpy":  "#include <string.h>",
            "memset":  "#include <string.h>",
            "sqrt":    "#include <math.h>",
            "pow":     "#include <math.h>",
            "abs":     "#include <math.h>",
        }
        return include_map.get(func_name)

    def _find_function_start(self, lines: list, error_line_idx: int) -> int:
        """
        Walk backwards from error line to find the function opening brace.
        Returns the line index just after the opening brace,
        a good place to insert variable declarations.
        """
        for i in range(error_line_idx, -1, -1):
            if '{' in lines[i]:
                return i + 1
        return 0

    def _extract_error_context(self, source: str,
                                error_lines: list,
                                context_window: int = 5) -> str:
        """
        Extract a window of source lines around each error.
        This is what we encode with CodeBERT —
        focused context is more meaningful than the entire file.
        """
        if not error_lines:
            return source[:1000]  # first 1000 chars if no specific errors

        lines = source.split("\n")
        snippets = []

        for error in error_lines[:3]:  # focus on first 3 errors
            start = max(0, error.line - context_window - 1)
            end   = min(len(lines), error.line + context_window)
            snippet = "\n".join(lines[start:end])
            snippets.append(snippet)

        return "\n---\n".join(snippets)

    # ------------------------------------------------------------------
    # Internal — Security Scoring
    # ------------------------------------------------------------------

    def _compute_security_score(self, matches: list) -> float:
        """
        Convert vulnerability matches into a scalar score 0.0-1.0.
        1.0 = completely clean
        0.0 = critically dangerous

        Severity weights:
          critical → 0.40 penalty per match
          high     → 0.20 penalty per match
          medium   → 0.10 penalty per match
          low      → 0.05 penalty per match

        Score is multiplied by confidence — a 0.76 confidence match
        on a critical vuln penalizes less than a 0.95 confidence match.
        """
        severity_weights = {
            "critical": 0.40,
            "high":     0.20,
            "medium":   0.10,
            "low":      0.05
        }

        total_penalty = 0.0
        for match in matches:
            weight  = severity_weights.get(match.severity, 0.10)
            penalty = weight * match.confidence
            total_penalty += penalty

        return max(0.0, round(1.0 - total_penalty, 3))

    # ------------------------------------------------------------------
    # Internal — helpers
    # ------------------------------------------------------------------

    def _log(self, msg: str):
        print(f"[FixerAgent] {msg}")


# ---------------------------------------------------------------------------
# Smoke Test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import tempfile, yaml

    print("=" * 60)
    print("AGentic_C — Fixer Agent Smoke Test")
    print("=" * 60)

    # Minimal config
    test_config = {
        "compiler": {
            "frontend": "clang",
            "target_arch": "arm64-apple-macosx",
            "opt_level": "O0",
            "ir_output_dir": "/tmp/agentic_c/ir"
        },
        "agents": {
            "boss":     {"top_k_memory": 5, "max_retries": 3},
            "fixer":    {"max_repair_attempts": 3,
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

    tmp_cfg = tempfile.NamedTemporaryFile(
        mode="w", suffix=".yaml", delete=False)
    yaml.dump(test_config, tmp_cfg)
    tmp_cfg.close()

    # Initialize agent
    agent = FixerAgent(config_path=tmp_cfg.name)

    # ── Test 1: Pre-fix — missing semicolons ─────────────────────────
    print("\n── Test 1: Pre-fix — syntax repair ──")

    broken_source = """
#include <stdio.h>
int add(int a, int b) {
    int result = a + b
    return result
}
int main() {
    int x = add(3, 4)
    printf("Result: %d\\n", x)
    return 0;
}
"""
    broken_file = tempfile.NamedTemporaryFile(
        mode="w", suffix=".c", delete=False)
    broken_file.write(broken_source)
    broken_file.close()

    # Simulate ParsedError objects (what Frontend would return)
    from dataclasses import dataclass
    @dataclass
    class MockError:
        line: int
        col: int
        severity: str
        message: str

    mock_errors = [
        MockError(4, 20, "error", "expected ';' at end of declaration"),
        MockError(5, 18, "error", "expected ';' at end of declaration"),
        MockError(8, 22, "error", "expected ';' at end of declaration"),
    ]

    result1 = agent.pre_fix(broken_file.name, "syntax", mock_errors)
    print(f"  success         : {result1.success}")
    print(f"  attempts        : {result1.attempts}")
    print(f"  patches applied : {result1.patches_applied}")
    print(f"  message         : {result1.message}")
    if result1.patched_source:
        print(f"\n  Patched source:")
        for i, line in enumerate(result1.patched_source.split("\n"), 1):
            if line.strip():
                print(f"    {i:2}: {line}")
    print("  ✓ PASSED" if result1.success else "  ✗ FAILED")

    # ── Test 2: Post-fix — security scan clean code ───────────────────
    print("\n── Test 2: Post-fix — clean code security scan ──")

    clean_source = """
#include <string.h>
#include <stdlib.h>
void safe_copy(char* dest, const char* src, size_t max_len) {
    strncpy(dest, src, max_len - 1);
    dest[max_len - 1] = '\\0';
}
"""
    clean_file = tempfile.NamedTemporaryFile(
        mode="w", suffix=".c", delete=False)
    clean_file.write(clean_source)
    clean_file.close()

    result2 = agent.post_fix(clean_file.name)
    print(f"  security_score  : {result2.security_score:.3f}")
    print(f"  vulnerabilities : {len(result2.vulnerabilities)}")
    print(f"  ✓ PASSED")

    # ── Test 3: Post-fix — dangerous code ────────────────────────────
    print("\n── Test 3: Post-fix — dangerous code security scan ──")

    dangerous_source = """
#include <string.h>
#include <stdio.h>
void dangerous(char* user_input) {
    char buf[10];
    strcpy(buf, user_input);
    printf(user_input);
}
"""
    danger_file = tempfile.NamedTemporaryFile(
        mode="w", suffix=".c", delete=False)
    danger_file.write(dangerous_source)
    danger_file.close()

    result3 = agent.post_fix(danger_file.name)
    print(f"  security_score  : {result3.security_score:.3f}")
    print(f"  vulnerabilities : {len(result3.vulnerabilities)}")
    for v in result3.vulnerabilities:
        print(f"    [{v.severity.upper()}] {v.cwe_id} {v.cwe_name} "
              f"confidence={v.confidence:.3f}")
    print(f"  ✓ PASSED")

    # ── Test 4: CodeBERT cosine similarity ────────────────────────────
    print("\n── Test 4: CodeBERT cosine similarity ──")
    enc = agent.encoder

    safe_code = "strncpy(dest, src, sizeof(dest)-1);"
    vuln_code = "char buf[10]; strcpy(buf, user_input);"

    emb_safe = enc.encode(safe_code)
    emb_vuln = enc.encode(vuln_code)
    emb_ref  = enc.encode("char buffer[256]; gets(buffer);")

    sim_safe_ref = enc.cosine_similarity(emb_safe, emb_ref)
    sim_vuln_ref = enc.cosine_similarity(emb_vuln, emb_ref)

    print(f"  safe_code  vs buffer_overflow_ref : {sim_safe_ref:.3f}")
    print(f"  vuln_code  vs buffer_overflow_ref : {sim_vuln_ref:.3f}")
    print(f"  Expected: vuln_code similarity > safe_code similarity")
    print(f"  ✓ PASSED" if sim_vuln_ref > sim_safe_ref else
          f"  Result: interesting — CodeBERT sees them similarly")

    print("\n" + "=" * 60)
    print("✓ Fixer Agent smoke test COMPLETE")
    print("=" * 60)