"""
AGentic_C — Experience Store
==============================
Stores and retrieves compilation experiences for the Boss Agent's
memory-informed planning.

Each experience record contains:
  - ir_embedding   : 256-dim float vector (from SimpleIREncoder)
  - source_path    : which file was compiled
  - plan_json      : the CompilationPlan that was applied
  - reward         : composite reward score (0.0 – 1.0)
  - hot_units      : which functions were classified HOT
  - anti_patterns  : LAP codes detected by the Fixer Agent
  - latency_before : ns estimate before optimisation
  - latency_after  : ns estimate after optimisation
  - passes_applied : list of LLVM passes that were applied
  - timestamp      : when this record was written
  - hft_mode       : was HFT chain active?

Two backends supported:
  PostgreSQL + pgvector  — production, full vector similarity search
  SQLite + numpy         — fallback, works with no extra setup

The Boss Agent calls:
  store.save(embedding, plan, reward, metadata)
  store.query_similar(embedding, top_k) → list of past experiences

On first run with PostgreSQL, the store creates the table and installs
the pgvector extension automatically.
"""

import os
import sys
import json
import time
import yaml
import sqlite3
import tempfile
import numpy as np
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Optional


# ---------------------------------------------------------------------------
# Experience record
# ---------------------------------------------------------------------------

@dataclass
class Experience:
    """One stored compilation experience."""
    id:              int             = 0
    source_path:     str             = ""
    ir_embedding:    list            = field(default_factory=list)   # 256-dim
    plan_json:       str             = "{}"
    reward:          float           = 0.0
    hot_units:       str             = "[]"   # JSON list of unit names
    anti_patterns:   str             = "[]"   # JSON list of LAP codes
    latency_before:  float           = 0.0
    latency_after:   float           = 0.0
    passes_applied:  str             = "[]"   # JSON list
    hft_mode:        bool            = True
    timestamp:       str             = ""

    def embedding_array(self) -> np.ndarray:
        return np.array(self.ir_embedding, dtype=np.float32)

    def similarity_to(self, query: np.ndarray) -> float:
        """Cosine similarity between this experience and a query embedding."""
        emb = self.embedding_array()
        norm_e = np.linalg.norm(emb)
        norm_q = np.linalg.norm(query)
        if norm_e == 0 or norm_q == 0:
            return 0.0
        return float(np.dot(emb, query) / (norm_e * norm_q))


# ---------------------------------------------------------------------------
# SQLite backend (always available, no extra setup)
# ---------------------------------------------------------------------------

class SQLiteStore:
    """
    Lightweight experience store backed by SQLite.
    No extra dependencies — works out of the box.
    Similarity search done in Python with numpy (cosine similarity).
    Good enough for development and course demo.
    For production, swap to PostgreSQLStore.
    """

    def __init__(self, db_path: str = "/tmp/agentic_c/experience_store.db"):
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        self.db_path = db_path
        self.conn    = sqlite3.connect(db_path, check_same_thread=False)
        self._init_schema()
        self._log(f"SQLite store ready: {db_path}")

    def _init_schema(self):
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS experiences (
                id              INTEGER PRIMARY KEY AUTOINCREMENT,
                source_path     TEXT,
                ir_embedding    TEXT,       -- JSON array of 256 floats
                plan_json       TEXT,
                reward          REAL,
                hot_units       TEXT,
                anti_patterns   TEXT,
                latency_before  REAL,
                latency_after   REAL,
                passes_applied  TEXT,
                hft_mode        INTEGER,
                timestamp       TEXT
            )
        """)
        self.conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_reward
            ON experiences(reward DESC)
        """)
        self.conn.commit()

    def save(self, embedding: np.ndarray, plan_json: str,
             reward: float, metadata: dict) -> int:
        """
        Saves one experience. Returns the new record id.
        Only saves if reward >= min_reward_threshold.
        """
        cur = self.conn.execute("""
            INSERT INTO experiences
            (source_path, ir_embedding, plan_json, reward,
             hot_units, anti_patterns, latency_before, latency_after,
             passes_applied, hft_mode, timestamp)
            VALUES (?,?,?,?,?,?,?,?,?,?,?)
        """, (
            metadata.get("source_path", ""),
            json.dumps(embedding.tolist()),
            plan_json,
            reward,
            json.dumps(metadata.get("hot_units", [])),
            json.dumps(metadata.get("anti_patterns", [])),
            metadata.get("latency_before", 0.0),
            metadata.get("latency_after", 0.0),
            json.dumps(metadata.get("passes_applied", [])),
            int(metadata.get("hft_mode", True)),
            datetime.utcnow().isoformat(),
        ))
        self.conn.commit()
        return cur.lastrowid

    def query_similar(self, query_embedding: np.ndarray,
                      top_k: int = 5,
                      min_reward: float = 0.0) -> list[Experience]:
        """
        Returns top_k most similar past experiences by cosine similarity.
        Filters by min_reward before similarity ranking.
        """
        rows = self.conn.execute("""
            SELECT id, source_path, ir_embedding, plan_json, reward,
                   hot_units, anti_patterns, latency_before, latency_after,
                   passes_applied, hft_mode, timestamp
            FROM experiences
            WHERE reward >= ?
            ORDER BY reward DESC
            LIMIT 200
        """, (min_reward,)).fetchall()

        if not rows:
            return []

        # Score by cosine similarity
        scored = []
        for row in rows:
            emb = np.array(json.loads(row[2]), dtype=np.float32)
            sim = self._cosine(query_embedding, emb)
            scored.append((sim, row))

        scored.sort(key=lambda x: x[0], reverse=True)
        top = scored[:top_k]

        results = []
        for sim, row in top:
            exp = Experience(
                id             = row[0],
                source_path    = row[1],
                ir_embedding   = json.loads(row[2]),
                plan_json      = row[3],
                reward         = row[4],
                hot_units      = row[5],
                anti_patterns  = row[6],
                latency_before = row[7],
                latency_after  = row[8],
                passes_applied = row[9],
                hft_mode       = bool(row[10]),
                timestamp      = row[11],
            )
            results.append(exp)

        return results

    def get_stats(self) -> dict:
        """Returns summary statistics about the store."""
        row = self.conn.execute("""
            SELECT COUNT(*), AVG(reward), MAX(reward),
                   AVG(latency_before), AVG(latency_after)
            FROM experiences
        """).fetchone()
        return {
            "total_experiences": row[0],
            "avg_reward":        round(row[1] or 0, 4),
            "max_reward":        round(row[2] or 0, 4),
            "avg_latency_before": round(row[3] or 0, 1),
            "avg_latency_after":  round(row[4] or 0, 1),
            "db_path":           self.db_path,
            "backend":           "sqlite",
        }

    def get_all(self, limit: int = 50) -> list[Experience]:
        """Returns most recent experiences, for inspection."""
        rows = self.conn.execute("""
            SELECT id, source_path, ir_embedding, plan_json, reward,
                   hot_units, anti_patterns, latency_before, latency_after,
                   passes_applied, hft_mode, timestamp
            FROM experiences
            ORDER BY id DESC
            LIMIT ?
        """, (limit,)).fetchall()

        return [
            Experience(
                id=r[0], source_path=r[1], ir_embedding=json.loads(r[2]),
                plan_json=r[3], reward=r[4], hot_units=r[5],
                anti_patterns=r[6], latency_before=r[7], latency_after=r[8],
                passes_applied=r[9], hft_mode=bool(r[10]), timestamp=r[11]
            )
            for r in rows
        ]

    def _cosine(self, a: np.ndarray, b: np.ndarray) -> float:
        na, nb = np.linalg.norm(a), np.linalg.norm(b)
        if na == 0 or nb == 0:
            return 0.0
        return float(np.dot(a, b) / (na * nb))

    def _log(self, msg: str):
        print(f"[ExperienceStore] {msg}")


# ---------------------------------------------------------------------------
# PostgreSQL + pgvector backend (production)
# ---------------------------------------------------------------------------

class PostgreSQLStore:
    """
    Production experience store backed by PostgreSQL + pgvector.
    Enables true vector similarity search (ANN) at scale.

    Requires:
      pip install psycopg2-binary pgvector
      PostgreSQL running with pgvector extension installed

    Schema uses pgvector's <-> operator for L2 distance similarity search,
    which is equivalent to cosine similarity on normalised vectors.
    """

    SCHEMA = """
        CREATE EXTENSION IF NOT EXISTS vector;

        CREATE TABLE IF NOT EXISTS experiences (
            id              SERIAL PRIMARY KEY,
            source_path     TEXT,
            ir_embedding    vector(256),
            plan_json       JSONB,
            reward          FLOAT,
            hot_units       JSONB,
            anti_patterns   JSONB,
            latency_before  FLOAT,
            latency_after   FLOAT,
            passes_applied  JSONB,
            hft_mode        BOOLEAN,
            timestamp       TIMESTAMPTZ DEFAULT NOW()
        );

        CREATE INDEX IF NOT EXISTS idx_experiences_embedding
        ON experiences USING ivfflat (ir_embedding vector_cosine_ops)
        WITH (lists = 100);

        CREATE INDEX IF NOT EXISTS idx_experiences_reward
        ON experiences (reward DESC);
    """

    def __init__(self, config: dict):
        mem = config.get("memory", {})
        self.host    = mem.get("host", "localhost")
        self.port    = mem.get("port", 5432)
        self.db      = mem.get("db", "agentic_c")
        self.dim     = mem.get("vector_dim", 256)
        self.conn    = None
        self._connect()

    def _connect(self):
        try:
            import psycopg2
            from pgvector.psycopg2 import register_vector
            self.conn = psycopg2.connect(
                host=self.host, port=self.port,
                dbname=self.db, user=os.getenv("PGUSER", "postgres"),
                password=os.getenv("PGPASSWORD", ""),
            )
            register_vector(self.conn)
            with self.conn.cursor() as cur:
                cur.execute(self.SCHEMA)
            self.conn.commit()
            self._log(f"PostgreSQL store ready: {self.host}:{self.port}/{self.db}")
        except Exception as e:
            raise RuntimeError(f"PostgreSQL connection failed: {e}") from e

    def save(self, embedding: np.ndarray, plan_json: str,
             reward: float, metadata: dict) -> int:
        with self.conn.cursor() as cur:
            cur.execute("""
                INSERT INTO experiences
                (source_path, ir_embedding, plan_json, reward,
                 hot_units, anti_patterns, latency_before, latency_after,
                 passes_applied, hft_mode)
                VALUES (%s, %s, %s::jsonb, %s, %s::jsonb, %s::jsonb,
                        %s, %s, %s::jsonb, %s)
                RETURNING id
            """, (
                metadata.get("source_path", ""),
                embedding,
                plan_json,
                reward,
                json.dumps(metadata.get("hot_units", [])),
                json.dumps(metadata.get("anti_patterns", [])),
                metadata.get("latency_before", 0.0),
                metadata.get("latency_after", 0.0),
                json.dumps(metadata.get("passes_applied", [])),
                metadata.get("hft_mode", True),
            ))
            new_id = cur.fetchone()[0]
        self.conn.commit()
        return new_id

    def query_similar(self, query_embedding: np.ndarray,
                      top_k: int = 5,
                      min_reward: float = 0.0) -> list[Experience]:
        """Uses pgvector cosine similarity index for fast ANN search."""
        with self.conn.cursor() as cur:
            cur.execute("""
                SELECT id, source_path, ir_embedding::text, plan_json::text,
                       reward, hot_units::text, anti_patterns::text,
                       latency_before, latency_after,
                       passes_applied::text, hft_mode, timestamp
                FROM experiences
                WHERE reward >= %s
                ORDER BY ir_embedding <=> %s
                LIMIT %s
            """, (min_reward, query_embedding, top_k))
            rows = cur.fetchall()

        return [
            Experience(
                id=r[0], source_path=r[1],
                ir_embedding=json.loads(r[2]),
                plan_json=r[3], reward=r[4],
                hot_units=r[5], anti_patterns=r[6],
                latency_before=r[7], latency_after=r[8],
                passes_applied=r[9], hft_mode=bool(r[10]),
                timestamp=str(r[11]),
            )
            for r in rows
        ]

    def get_stats(self) -> dict:
        with self.conn.cursor() as cur:
            cur.execute("""
                SELECT COUNT(*), AVG(reward), MAX(reward),
                       AVG(latency_before), AVG(latency_after)
                FROM experiences
            """)
            row = cur.fetchone()
        return {
            "total_experiences": row[0],
            "avg_reward":        round(float(row[1] or 0), 4),
            "max_reward":        round(float(row[2] or 0), 4),
            "avg_latency_before": round(float(row[3] or 0), 1),
            "avg_latency_after":  round(float(row[4] or 0), 1),
            "backend":           "postgresql+pgvector",
            "host":              f"{self.host}:{self.port}/{self.db}",
        }

    def _log(self, msg: str):
        print(f"[ExperienceStore] {msg}")


# ---------------------------------------------------------------------------
# ExperienceStore — public interface
# Auto-selects backend based on config and availability
# ---------------------------------------------------------------------------

class ExperienceStore:
    """
    Public interface to the experience store.
    Auto-selects PostgreSQL if available, falls back to SQLite.

    Usage:
        store = ExperienceStore(config)
        store.save(embedding, plan, reward, metadata)
        past = store.query_similar(embedding, top_k=5)
    """

    def __init__(self, config: dict = None, db_path: str = None):
        self.config        = config or {}
        self.min_reward    = self.config.get("memory", {}).get(
                             "min_reward_threshold", 0.75)
        self._backend      = None
        self._backend_name = "none"
        self._init_backend(db_path)

    def _init_backend(self, db_path: str = None):
        """Try PostgreSQL first, fall back to SQLite."""
        # Try PostgreSQL
        mem = self.config.get("memory", {})
        if mem.get("host") and mem.get("db"):
            try:
                self._backend      = PostgreSQLStore(self.config)
                self._backend_name = "postgresql"
                return
            except Exception as e:
                print(f"[ExperienceStore] PostgreSQL unavailable ({e}), "
                      f"falling back to SQLite.")

        # SQLite fallback
        path = db_path or mem.get(
            "sqlite_path", "/tmp/agentic_c/experience_store.db"
        )
        self._backend      = SQLiteStore(path)
        self._backend_name = "sqlite"

    def save(self, embedding: np.ndarray,
             plan,                          # CompilationPlan
             reward: float,
             metadata: dict = None) -> bool:
        """
        Saves one compilation experience.
        Only stores if reward >= min_reward_threshold.

        Args:
            embedding:  256-dim IR embedding (from SimpleIREncoder)
            plan:       CompilationPlan dataclass
            reward:     composite reward score
            metadata:   dict with source_path, hot_units, anti_patterns,
                        latency_before, latency_after, passes_applied, hft_mode

        Returns True if stored, False if below threshold.
        """
        if reward < self.min_reward:
            print(f"[ExperienceStore] Skipped (reward={reward:.3f} "
                  f"< threshold={self.min_reward})")
            return False

        metadata = metadata or {}

        # Serialise plan to JSON
        try:
            plan_dict = {
                "ir_tuner_budget":   getattr(plan, "ir_tuner_budget", 35),
                "hw_tuner_budget":   getattr(plan, "hw_tuner_budget", 14),
                "hft_chain_active":  getattr(plan, "hft_chain_active", True),
                "ir_tuner_directive":getattr(plan, "ir_tuner_directive", ""),
                "based_on_memory":   getattr(plan, "based_on_memory", False),
                "confidence":        getattr(plan, "confidence", 0.5),
                "hot_units": [
                    getattr(u, "unit_name", str(u))
                    for u in getattr(plan, "hot_units", [])
                ],
                "cold_units": [
                    getattr(u, "unit_name", str(u))
                    for u in getattr(plan, "cold_units", [])
                ],
            }
            plan_json = json.dumps(plan_dict)
        except Exception:
            plan_json = "{}"

        try:
            new_id = self._backend.save(embedding, plan_json, reward, metadata)
            print(f"[ExperienceStore] Saved experience #{new_id} "
                  f"(reward={reward:.3f}, backend={self._backend_name})")
            return True
        except Exception as e:
            print(f"[ExperienceStore] Save failed: {e}")
            return False

    def query_similar(self, embedding: np.ndarray,
                      top_k: int = 5) -> list[dict]:
        """
        Returns top_k most similar past experiences as dicts.
        Each dict contains plan_json, reward, hot_units, passes_applied, etc.
        Ready to be used directly by Boss Agent's _build_plan().
        """
        try:
            experiences = self._backend.query_similar(
                embedding, top_k=top_k, min_reward=self.min_reward
            )
            return [self._to_dict(e) for e in experiences]
        except Exception as e:
            print(f"[ExperienceStore] Query failed: {e}")
            return []

    def get_stats(self) -> dict:
        """Returns store statistics — call this to see what's accumulated."""
        try:
            return self._backend.get_stats()
        except Exception as e:
            return {"error": str(e)}

    def get_recent(self, limit: int = 10) -> list[dict]:
        """Returns most recently stored experiences."""
        try:
            experiences = self._backend.get_all(limit=limit)
            return [self._to_dict(e) for e in experiences]
        except Exception as e:
            return []

    def _to_dict(self, exp: Experience) -> dict:
        """Converts Experience to the dict format Boss Agent expects."""
        try:
            plan = json.loads(exp.plan_json)
        except Exception:
            plan = {}

        return {
            "id":              exp.id,
            "source_path":     exp.source_path,
            "reward":          exp.reward,
            "plan":            plan,
            "ir_tuner_budget": plan.get("ir_tuner_budget", 35),
            "hw_tuner_budget": plan.get("hw_tuner_budget", 14),
            "hot_units":       json.loads(exp.hot_units) if isinstance(exp.hot_units, str) else exp.hot_units,
            "anti_patterns":   json.loads(exp.anti_patterns) if isinstance(exp.anti_patterns, str) else exp.anti_patterns,
            "passes_applied":  json.loads(exp.passes_applied) if isinstance(exp.passes_applied, str) else exp.passes_applied,
            "latency_before":  exp.latency_before,
            "latency_after":   exp.latency_after,
            "latency_delta":   round(exp.latency_before - exp.latency_after, 2),
            "hft_mode":        exp.hft_mode,
            "timestamp":       exp.timestamp,
            "similarity":      0.0,   # populated by query_similar caller
        }

    @property
    def backend(self) -> str:
        return self._backend_name


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=" * 68)
    print("AGentic_C — Experience Store Smoke Test")
    print("=" * 68)

    # Use SQLite for testing (no PostgreSQL needed)
    store = ExperienceStore(config={
        "memory": {
            "min_reward_threshold": 0.70,
            "sqlite_path": "/tmp/agentic_c/test_store.db",
        }
    })

    print(f"\n  Backend: {store.backend}")

    # ── Generate fake embeddings ──────────────────────────────────────
    rng = np.random.default_rng(42)

    def fake_embedding(seed_offset=0):
        v = rng.random(256).astype(np.float32) + seed_offset * 0.1
        return v / np.linalg.norm(v)   # normalise

    # ── Test 1: Save below threshold ─────────────────────────────────
    print("\n── Test 1: Save below threshold ──")

    class FakePlan:
        ir_tuner_budget   = 35
        hw_tuner_budget   = 14
        hft_chain_active  = True
        ir_tuner_directive= ""
        based_on_memory   = False
        confidence        = 0.5
        hot_units         = []
        cold_units        = []

    stored = store.save(
        fake_embedding(), FakePlan(), reward=0.50,
        metadata={"source_path": "low_reward.cpp", "hft_mode": True}
    )
    assert not stored, "Should not store below threshold"
    print("  ✓ PASSED — low reward correctly rejected")

    # ── Test 2: Save above threshold ─────────────────────────────────
    print("\n── Test 2: Save above threshold ──")

    experiences_to_store = [
        {
            "source_path":   "on_market_data.cpp",
            "reward":        0.88,
            "latency_before": 340.0,
            "latency_after":  212.0,
            "hot_units":     ["on_market_data", "evaluate_signal"],
            "anti_patterns": ["LAP-001", "LAP-004"],
            "passes_applied": ["mem2reg", "sroa", "loop-vectorize"],
            "hft_mode":      True,
        },
        {
            "source_path":   "risk_check.cpp",
            "reward":        0.92,
            "latency_before": 150.0,
            "latency_after":  98.0,
            "hot_units":     ["check_risk"],
            "anti_patterns": [],
            "passes_applied": ["mem2reg", "inline", "licm"],
            "hft_mode":      True,
        },
        {
            "source_path":   "order_submit.cpp",
            "reward":        0.76,
            "latency_before": 280.0,
            "latency_after":  231.0,
            "hot_units":     ["submit_order"],
            "anti_patterns": ["LAP-006"],
            "passes_applied": ["inline", "always-inline", "simplifycfg"],
            "hft_mode":      True,
        },
    ]

    ids = []
    for meta in experiences_to_store:
        reward = meta.pop("reward")
        emb = fake_embedding(seed_offset=len(ids))
        ok = store.save(emb, FakePlan(), reward=reward, metadata=meta)
        assert ok, f"Should have stored {meta['source_path']}"
        ids.append(ok)

    print(f"  ✓ PASSED — stored {len(experiences_to_store)} experiences")

    # ── Test 3: Query similar ─────────────────────────────────────────
    print("\n── Test 3: Query similar ──")

    # Query with embedding similar to first one
    query_emb = fake_embedding(seed_offset=0)
    results = store.query_similar(query_emb, top_k=3)

    assert len(results) > 0, "Should return results"
    assert all("reward" in r for r in results), "All results should have reward"
    assert all(r["reward"] >= 0.70 for r in results), "All above threshold"

    print(f"  ✓ PASSED — retrieved {len(results)} similar experiences:")
    for r in results:
        delta = r["latency_delta"]
        print(f"    #{r['id']:2d}  {r['source_path']:<25} "
              f"reward={r['reward']:.2f}  "
              f"Δlat={delta:.0f}ns  "
              f"passes={r['passes_applied'][:3]}")

    # ── Test 4: Stats ─────────────────────────────────────────────────
    print("\n── Test 4: Store statistics ──")
    stats = store.get_stats()
    assert stats["total_experiences"] >= 3, "Should have at least 3 records"
    print(f"  ✓ PASSED")
    print(f"    Total records:    {stats['total_experiences']}")
    print(f"    Avg reward:       {stats['avg_reward']:.4f}")
    print(f"    Max reward:       {stats['max_reward']:.4f}")
    print(f"    Avg Δlatency:     "
          f"{stats['avg_latency_before']:.0f}ns → "
          f"{stats['avg_latency_after']:.0f}ns")
    print(f"    Backend:          {stats['backend']}")

    # ── Test 5: Recent experiences ────────────────────────────────────
    print("\n── Test 5: Recent experiences ──")
    recent = store.get_recent(limit=5)
    assert len(recent) > 0, "Should have recent experiences"
    print(f"  ✓ PASSED — {len(recent)} recent records:")
    for r in recent:
        print(f"    {r['timestamp'][:19]}  {r['source_path']:<25} "
              f"reward={r['reward']:.2f}")

    # ── Test 6: Boss Agent integration format ─────────────────────────
    print("\n── Test 6: Boss Agent integration ──")
    past = store.query_similar(fake_embedding(), top_k=1)
    if past:
        best = max(past, key=lambda x: x["reward"])
        has_plan    = "plan" in best
        has_passes  = "passes_applied" in best
        has_lat     = "latency_delta" in best
        if has_plan and has_passes and has_lat:
            print(f"  ✓ PASSED — Boss Agent can use result:")
            print(f"    best reward:      {best['reward']:.3f}")
            print(f"    ir_tuner_budget:  {best['ir_tuner_budget']}")
            print(f"    passes:           {best['passes_applied']}")
            print(f"    latency delta:    {best['latency_delta']:.0f}ns")
        else:
            print(f"  ✗ FAILED — missing fields: {best.keys()}")

    # Cleanup test db
    import os as _os
    try:
        _os.unlink("/tmp/agentic_c/test_store.db")
    except Exception:
        pass

    print()
    print("=" * 68)
    print("✓ Experience Store smoke test PASSED")
    print("=" * 68)
    print()
    print("── PostgreSQL setup (when ready for production) ──")
    print("  brew install postgresql")
    print("  brew services start postgresql")
    print("  createdb agentic_c")
    print("  psql agentic_c -c 'CREATE EXTENSION vector;'")
    print("  pip install psycopg2-binary pgvector --break-system-packages")
    print("  # Then set PGUSER and PGPASSWORD env vars")
    print("  # ExperienceStore will auto-detect and switch to PostgreSQL")
