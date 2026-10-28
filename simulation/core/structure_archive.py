"""A durable record of every structure a search evaluated, not a candidate library.

A 100 generation run evaluates around two thousand structures and reports five.
The other 2084 currently exist only in memory for the length of a session and
inside a JSON blob afterwards, so they cannot be queried, compared or reused.

Two reasons to keep them, one practical and one about honesty.

The practical one is measured: re-scoring 2093 archived structures against a new
model takes 99 seconds, against 6834 seconds to repeat the search that produced
them, a factor of 69. Since expanding the training set is the project's top
priority, and expanding it invalidates every score, an archive turns a two hour
re-run into a two minute re-rank.

The honest one is that the explored space is a result. The generator reaches 734
distinct Murcko scaffolds while selection pulls back toward the 43 training
compounds; that tension is the most informative thing a run produces and it is
currently invisible.

What this is not, and the schema enforces it. Every row is typed `search_trace`,
carries the reason it was not selected, carries the model versions that produced
its scores so stale numbers are identifiable, and carries a caveat that travels
into every export. A table of machine generated molecules with scores looks like
a screening library. It is a log of what an algorithm tried.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import json
import sqlite3

DEFAULT_DB_PATH: Path = Path(__file__).resolve().parents[2] / "experiments" / "evocure.sqlite3"

RECORD_TYPE: str = "search_trace"

EXPORT_CAVEAT: str = (
    "SEARCH TRACE, NOT A CANDIDATE LIBRARY. These structures are what a generative "
    "search evaluated, including the great majority it rejected. Scores come from "
    "models whose uncertainty exceeds the differences between rows: the potency model's "
    "95% conformal interval is a factor of about 1330 on IC50. A high score here means "
    "an algorithm ranked it highly, not that the compound is active, synthesisable, "
    "safe or new. Around half of the structurally distant entries trip published "
    "medicinal chemistry alerts. Nothing here has been made or tested."
)


@dataclass
class RescoreOutcome:
    """What changed when the archive was scored against a different model."""

    total: int
    rescored: int
    unchanged: int
    improved: int
    worsened: int
    rank_churn_top_50: int
    previous_model: str
    new_model: str
    comparable_objective: bool = True

    def to_dict(self) -> dict[str, Any]:
        if self.comparable_objective:
            reading = (
                f"{self.rank_churn_top_50} of the previous top 50 are no longer in the "
                f"top 50. Because the objective is unchanged, churn here is model drift: "
                f"high churn means the ranking was driven by the model rather than by the "
                f"chemistry, which is what a wide conformal interval predicts."
            )
        else:
            reading = (
                f"{self.rank_churn_top_50} of the previous top 50 are no longer in the "
                f"top 50, but the objective also changed, so this number does not measure "
                f"model drift. Scoring on different criteria reorders the archive by "
                f"construction. Compare like with like before reading anything into churn."
            )
        return {
            "total": self.total,
            "rescored": self.rescored,
            "unchanged": self.unchanged,
            "improved": self.improved,
            "worsened": self.worsened,
            "rank_churn_top_50": self.rank_churn_top_50,
            "previous_model": self.previous_model,
            "new_model": self.new_model,
            "comparable_objective": self.comparable_objective,
            "interpretation": reading,
        }


class StructureArchive:
    """Append-only store of evaluated structures, keyed by InChIKey."""

    def __init__(self, db_path: str | Path | None = None) -> None:
        self.db_path = Path(db_path) if db_path else DEFAULT_DB_PATH
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self.db_path), timeout=30.0)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL;")
        conn.execute("PRAGMA synchronous=NORMAL;")
        return conn

    def _init_db(self) -> None:
        with self._connect() as conn:
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS structures (
                    inchikey TEXT PRIMARY KEY,
                    canonical_smiles TEXT NOT NULL,
                    murcko_scaffold TEXT,
                    formula TEXT,
                    mol_weight REAL,
                    logp REAL,
                    qed REAL,
                    record_type TEXT NOT NULL DEFAULT 'search_trace',
                    first_experiment_id TEXT,
                    first_seen_utc TEXT NOT NULL,
                    origin_generation INTEGER,
                    parent_ids_json TEXT,
                    transformation_type TEXT,
                    is_seed INTEGER NOT NULL DEFAULT 0,
                    objectives_json TEXT,
                    composite_utility REAL,
                    selected INTEGER NOT NULL DEFAULT 0,
                    not_selected_because TEXT,
                    potency_provenance TEXT,
                    pic50 REAL,
                    ic50_um REAL,
                    in_applicability_domain INTEGER,
                    medchem_json TEXT,
                    model_versions_json TEXT,
                    scored_at_utc TEXT,
                    caveat TEXT NOT NULL,
                    CHECK (record_type = 'search_trace')
                );

                CREATE INDEX IF NOT EXISTS idx_structures_utility
                    ON structures(composite_utility DESC);
                CREATE INDEX IF NOT EXISTS idx_structures_scaffold
                    ON structures(murcko_scaffold);
                CREATE INDEX IF NOT EXISTS idx_structures_experiment
                    ON structures(first_experiment_id);
                """
            )

    # ------------------------------------------------------------------ writing

    def record_search(
        self,
        result: dict[str, Any],
        experiment_id: str = "",
        model_versions: dict[str, Any] | None = None,
    ) -> dict[str, int]:
        """Archive every structure a pipeline run evaluated.

        Takes the pipeline's result dict directly, so callers do not reshape data
        and risk dropping the reason a structure was rejected.
        """
        from rdkit import Chem
        from rdkit.Chem.Scaffolds import MurckoScaffold

        evaluated = result.get("evaluated_candidates", []) or []
        selected = {c.get("compound_id") for c in result.get("top_candidates", []) or []}
        now = datetime.now(timezone.utc).isoformat()
        versions = json.dumps(model_versions or {}, sort_keys=True)

        inserted, updated, skipped = 0, 0, 0
        with self._connect() as conn:
            for item in evaluated:
                smiles = item.get("canonical_smiles")
                mol = Chem.MolFromSmiles(smiles) if smiles else None
                if mol is None:
                    skipped += 1
                    continue

                inchikey = Chem.MolToInchiKey(mol)
                is_selected = item.get("compound_id") in selected
                # Computed here rather than trusted from the caller: the pipeline's
                # candidate summary does not carry it, and scaffold count is the
                # archive's most informative statistic.
                scaffold = item.get("murcko_scaffold")
                if not scaffold:
                    try:
                        scaffold = MurckoScaffold.MurckoScaffoldSmiles(mol=mol)
                    except Exception:
                        scaffold = ""
                row = {
                    "inchikey": inchikey,
                    "canonical_smiles": Chem.MolToSmiles(mol, isomericSmiles=True),
                    "murcko_scaffold": scaffold,
                    "formula": item.get("formula"),
                    "mol_weight": item.get("mol_weight"),
                    "logp": item.get("logp"),
                    "qed": item.get("qed"),
                    "record_type": RECORD_TYPE,
                    "first_experiment_id": experiment_id,
                    "first_seen_utc": now,
                    "origin_generation": item.get("origin_generation"),
                    "parent_ids_json": json.dumps(item.get("parent_ids") or []),
                    "transformation_type": item.get("transformation_type"),
                    "is_seed": int(bool(item.get("is_seed"))),
                    "objectives_json": json.dumps(item.get("objectives") or {}, sort_keys=True),
                    "composite_utility": item.get("composite_utility"),
                    "selected": int(is_selected),
                    "not_selected_because": None if is_selected else _rejection_reason(item),
                    "potency_provenance": item.get("potency_provenance"),
                    "pic50": item.get("pic50"),
                    "ic50_um": item.get("ic50_um"),
                    "in_applicability_domain": int(bool(item.get("in_applicability_domain"))),
                    "medchem_json": json.dumps(item.get("medchem")) if item.get("medchem") else None,
                    "model_versions_json": versions,
                    "scored_at_utc": now,
                    "caveat": EXPORT_CAVEAT,
                }

                existing = conn.execute(
                    "SELECT composite_utility FROM structures WHERE inchikey = ?",
                    (inchikey,)).fetchone()
                if existing is None:
                    conn.execute(
                        f"INSERT INTO structures ({','.join(row)}) "
                        f"VALUES ({','.join(':' + k for k in row)})", row)
                    inserted += 1
                else:
                    # Keep the first sighting, refresh the scores.
                    row.pop("first_experiment_id")
                    row.pop("first_seen_utc")
                    conn.execute(
                        "UPDATE structures SET " +
                        ",".join(f"{k} = :{k}" for k in row if k != "inchikey") +
                        " WHERE inchikey = :inchikey", row)
                    updated += 1

        return {"inserted": inserted, "updated": updated, "skipped_invalid": skipped,
                "total_evaluated": len(evaluated)}

    # ------------------------------------------------------------------ reading

    def count(self) -> int:
        with self._connect() as conn:
            return int(conn.execute("SELECT COUNT(*) FROM structures").fetchone()[0])

    def query(
        self,
        limit: int = 50,
        min_utility: float | None = None,
        scaffold: str | None = None,
        exclude_alerts: bool = False,
        only_unselected: bool = False,
    ) -> list[dict[str, Any]]:
        clauses, params = [], []
        if min_utility is not None:
            clauses.append("composite_utility >= ?")
            params.append(min_utility)
        if scaffold:
            clauses.append("murcko_scaffold = ?")
            params.append(scaffold)
        if only_unselected:
            clauses.append("selected = 0")
        where = (" WHERE " + " AND ".join(clauses)) if clauses else ""

        with self._connect() as conn:
            rows = conn.execute(
                f"SELECT * FROM structures{where} "
                f"ORDER BY composite_utility DESC LIMIT ?", (*params, int(limit) * 4)
            ).fetchall()

        out = []
        for row in rows:
            record = dict(row)
            if exclude_alerts and record.get("medchem_json"):
                medchem = json.loads(record["medchem_json"])
                if medchem.get("alert_count", 0) > 0:
                    continue
            out.append(record)
            if len(out) >= limit:
                break
        return out

    def summary(self) -> dict[str, Any]:
        """What the search explored, which the report shows only five rows of."""
        with self._connect() as conn:
            total = conn.execute("SELECT COUNT(*) FROM structures").fetchone()[0]
            if not total:
                return {"total": 0, "caveat": EXPORT_CAVEAT}
            scaffolds = conn.execute(
                "SELECT COUNT(DISTINCT murcko_scaffold) FROM structures").fetchone()[0]
            selected = conn.execute(
                "SELECT COUNT(*) FROM structures WHERE selected = 1").fetchone()[0]
            seeds = conn.execute(
                "SELECT COUNT(*) FROM structures WHERE is_seed = 1").fetchone()[0]
            experiments = conn.execute(
                "SELECT COUNT(DISTINCT first_experiment_id) FROM structures").fetchone()[0]
            with_medchem = conn.execute(
                "SELECT COUNT(*) FROM structures WHERE medchem_json IS NOT NULL").fetchone()[0]
            reasons = conn.execute(
                "SELECT not_selected_because, COUNT(*) AS n FROM structures "
                "WHERE not_selected_because IS NOT NULL "
                "GROUP BY not_selected_because ORDER BY n DESC LIMIT 10").fetchall()

        return {
            "total": int(total),
            "unique_scaffolds": int(scaffolds),
            "selected_for_reporting": int(selected),
            "seed_compounds": int(seeds),
            "contributing_experiments": int(experiments),
            "with_medchem_assessment": int(with_medchem),
            "rejection_reasons": {r["not_selected_because"]: int(r["n"]) for r in reasons},
            "reported_fraction": round(selected / total, 5) if total else 0.0,
            "record_type": RECORD_TYPE,
            "caveat": EXPORT_CAVEAT,
        }

    def export(self, path: str | Path, limit: int = 1000) -> dict[str, Any]:
        """Write a JSON export. The caveat is a required field, not a footnote."""
        rows = self.query(limit=limit)
        payload = {
            "record_type": RECORD_TYPE,
            "caveat": EXPORT_CAVEAT,
            "exported_utc": datetime.now(timezone.utc).isoformat(),
            "summary": self.summary(),
            "structures": rows,
        }
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        return {"path": str(destination), "rows": len(rows)}

    # ----------------------------------------------------------------- rescoring

    def rescore(
        self,
        score_fn: Any,
        new_model_version: str,
        limit: int | None = None,
        comparable_objective: bool = True,
    ) -> RescoreOutcome:
        """Re-score archived structures against a new model.

        score_fn takes a SMILES and returns either None, to leave a row untouched,
        or a dict with any of pic50, ic50_um, composite_utility and objectives.

        The point of the archive: this is 69 times cheaper than repeating the
        search that produced these structures.
        """
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT inchikey, canonical_smiles, composite_utility, model_versions_json "
                "FROM structures ORDER BY composite_utility DESC"
                + (f" LIMIT {int(limit)}" if limit else "")
            ).fetchall()

        previous_order = [r["inchikey"] for r in rows[:50]]
        previous_model = ""
        if rows:
            try:
                previous_model = json.loads(rows[0]["model_versions_json"] or "{}").get(
                    "qsar", "unrecorded")
            except ValueError:
                previous_model = "unrecorded"

        now = datetime.now(timezone.utc).isoformat()
        rescored = improved = worsened = unchanged = 0
        updates = []

        for row in rows:
            scored = score_fn(row["canonical_smiles"])
            if not scored:
                continue
            rescored += 1
            before = row["composite_utility"]
            after = scored.get("composite_utility", before)
            if before is None or after is None or abs(after - before) < 1e-9:
                unchanged += 1
            elif after > before:
                improved += 1
            else:
                worsened += 1
            updates.append({
                "inchikey": row["inchikey"],
                "composite_utility": after,
                "pic50": scored.get("pic50"),
                "ic50_um": scored.get("ic50_um"),
                "objectives_json": json.dumps(scored.get("objectives") or {}, sort_keys=True),
                "model_versions_json": json.dumps({"qsar": new_model_version}, sort_keys=True),
                "scored_at_utc": now,
            })

        with self._connect() as conn:
            conn.executemany(
                "UPDATE structures SET composite_utility = :composite_utility, "
                "pic50 = :pic50, ic50_um = :ic50_um, objectives_json = :objectives_json, "
                "model_versions_json = :model_versions_json, scored_at_utc = :scored_at_utc "
                "WHERE inchikey = :inchikey", updates)
            new_rows = conn.execute(
                "SELECT inchikey FROM structures ORDER BY composite_utility DESC LIMIT 50"
            ).fetchall()

        new_order = [r["inchikey"] for r in new_rows]
        churn = len(set(previous_order) - set(new_order))

        return RescoreOutcome(
            total=len(rows), rescored=rescored, unchanged=unchanged,
            improved=improved, worsened=worsened, rank_churn_top_50=churn,
            previous_model=previous_model or "unrecorded", new_model=new_model_version,
            comparable_objective=comparable_objective,
        )


def _rejection_reason(item: dict[str, Any]) -> str:
    """Why a structure did not reach the report. Stored so the archive is readable."""
    objectives = item.get("objectives") or {}
    medchem = item.get("medchem") or {}

    if not item.get("in_applicability_domain", True):
        return "outside the model's applicability domain"
    if medchem.get("alert_count", 0) > 0:
        families = ", ".join(medchem.get("alert_families", [])) or "unspecified"
        return f"medicinal chemistry alerts ({families})"
    if objectives.get("predicted_pdac_activity", 1.0) < 0.3:
        return "low predicted activity"
    if objectives.get("low_toxicological_risk", 1.0) < 0.5:
        return "high computed toxicological risk"
    if objectives.get("qed_druglikeness", 1.0) < 0.3:
        return "low drug likeness"
    return "ranked below the reporting cut, no single disqualifying criterion"
