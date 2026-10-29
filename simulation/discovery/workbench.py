"""Local structural reference comparisons and bounded evolution search plans."""
from __future__ import annotations

from functools import lru_cache
import json
import re

from rdkit import Chem, DataStructs
from rdkit.Chem import rdFingerprintGenerator
from rdkit.Chem import rdDepictor

from simulation.chemistry.engine import ChemistryEngine
from simulation.chemistry.library import COMPOUND_LIBRARY
from simulation.core.contracts import InterfaceMode
from simulation.core.paths import reference_data_dir


@lru_cache(maxsize=2048)
def molecule_layout(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return {"atoms": [], "bonds": []}
    rdDepictor.Compute2DCoords(mol)
    conformer = mol.GetConformer()
    positions = [(conformer.GetAtomPosition(i).x, conformer.GetAtomPosition(i).y) for i in range(mol.GetNumAtoms())]
    extent = max((max(abs(x), abs(y)) for x, y in positions), default=1.0)
    return {"atoms": [[round(x / max(1.0, extent) * 35, 3), round(y / max(1.0, extent) * 35, 3), mol.GetAtomWithIdx(i).GetSymbol()] for i, (x, y) in enumerate(positions)],
            "bonds": [[bond.GetBeginAtomIdx(), bond.GetEndAtomIdx(), bond.GetBondTypeAsDouble()] for bond in mol.GetBonds()]}


@lru_cache(maxsize=1)
def drug_references():
    chemistry = ChemistryEngine()
    generator = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
    result = []
    seen = set()
    for entry in COMPOUND_LIBRARY:
        if entry.get("category") not in {"pdac_reference", "oncology", "antibiotic", "metabolic", "control"}:
            continue
        record = chemistry.resolve_compound_input(entry["name"])
        if record.canonical_smiles in seen:
            continue
        seen.add(record.canonical_smiles)
        mol = Chem.MolFromSmiles(record.canonical_smiles)
        result.append((entry, record, generator.GetFingerprint(mol)))
    return result


def compare_drugs(compound, limit=5):
    mol = Chem.MolFromSmiles(compound.canonical_smiles)
    if mol is None:
        raise ValueError("Candidate has no valid molecular structure")
    generator = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
    fingerprint = generator.GetFingerprint(mol)
    canonical = Chem.MolToSmiles(mol, isomericSmiles=True)
    connectivity = Chem.MolToSmiles(mol, isomericSmiles=False)
    matches = []
    for entry, reference, fp in drug_references():
        ref_mol = Chem.MolFromSmiles(reference.canonical_smiles)
        exact = canonical == Chem.MolToSmiles(ref_mol, isomericSmiles=True)
        same_connectivity = connectivity == Chem.MolToSmiles(ref_mol, isomericSmiles=False)
        descriptors = ("mol_weight", "logp", "tpsa", "hbd", "hba", "rotatable_bonds", "qed")
        matches.append({
            "name": entry["name"], "compound_id": entry["compound_id"],
            "category": entry["category"], "provenance": entry["provenance"],
            "canonical_smiles": reference.canonical_smiles, "formula": reference.formula,
            "svg_2d": reference.svg_2d,
            "similarity": round(float(DataStructs.TanimotoSimilarity(fingerprint, fp)), 4),
            "exact_match": exact, "same_connectivity": same_connectivity,
            "identity": "Exact structure" if exact else ("Connectivity matches; stereochemistry differs or is unspecified" if same_connectivity else "Similar structure only"),
            "descriptors": {key: getattr(reference, key) for key in descriptors},
            "candidate_minus_reference": {key: round(getattr(compound, key) - getattr(reference, key), 4) for key in descriptors},
        })
    matches.sort(key=lambda m: (m["exact_match"], m["similarity"]), reverse=True)
    exact = next((m["name"] for m in matches if m["exact_match"]), None)
    return {
        "candidate": {"compound_id": compound.compound_id, "name": compound.name,
                      "canonical_smiles": canonical, "formula": compound.formula, "svg_2d": compound.svg_2d,
                      "descriptors": {key: getattr(compound, key) for key in ("mol_weight", "logp", "tpsa", "hbd", "hba", "rotatable_bonds", "qed")}},
        "exact_reference_name": exact, "reference_count": len(matches),
        "matches": matches[:max(1, min(20, limit))],
        "method": "RDKit Morgan fingerprint, radius 2, 2048 bits; Tanimoto similarity. Exact identity uses canonical isomeric SMILES.",
        "scope": "Curated local reference drug library only; absence of a match does not establish novelty. Similarity does not demonstrate efficacy, safety, or a cure.",
        "artificial_complex": "." in canonical,
    }


def plan_guided_search(prompt, assistant=None):
    """LLM proposes categorical priorities; executable settings stay within bounds."""
    text = prompt.lower()
    count_match = re.search(r"\b(\d{1,2})\b", text)
    count = max(1, min(20, int(count_match.group(1)))) if count_match else 5
    priorities = []
    if any(word in text for word in ("toxic", "safe", "segur", "toxicidade")):
        priorities.append("safety")
    if any(word in text for word in ("drug-like", "druglike", "qed", "balanced", "equilibr")):
        priorities.append("druglikeness")
    if any(word in text for word in ("active", "poten", "cancer", "cancro", "cure", "cura")):
        priorities.append("activity")
    disease = "PDAC"
    for terms, name in [(("nsclc", "lung"), "NSCLC_KRAS"), (("amr", "bacter", "antibiotic"), "AMR_BACTERIAL"), (("masld", "metabol", "liver"), "METABOLIC_MASLD")]:
        if any(term in text for term in terms):
            disease = name
    allowed = {"safety", "druglikeness", "activity", "selectivity", "domain_confidence"}
    provider = "local_rule_based_planner"
    if assistant and assistant.is_local_server_running():
        question = ("Convert this research objective into a search policy, not medical claims. Return only JSON with priorities (an array chosen from safety, druglikeness, activity, selectivity, domain_confidence). Do not give molecules or numerical efficacy. Objective: " + prompt)
        answer = assistant._query_llama_server(question, {"task": "bounded_evolution_planning"}, InterfaceMode.LAB)
        try:
            raw = json.loads(answer[answer.index("{"):answer.rindex("}") + 1])
            suggested = [p for p in raw.get("priorities", []) if p in allowed]
            if suggested:
                priorities = list(dict.fromkeys(priorities + suggested))
                provider = "local_gemma_planner"
        except (ValueError, TypeError, AttributeError):
            pass
    keys = {"activity": "predicted_pdac_activity", "safety": "low_toxicological_risk",
            "druglikeness": "qed_druglikeness", "selectivity": "selectivity_index",
            "domain_confidence": "applicability_domain_confidence"}
    weights = {key: 1.0 for key in keys.values()}
    for priority in priorities:
        weights[keys[priority]] = 2.0
    if disease != "PDAC":
        weights["predicted_pdac_activity"] = 0.0
        weights["selectivity_index"] = 0.0
    explicit_seeds = [e["name"] for e in COMPOUND_LIBRARY if re.search(r"(?<![a-z0-9])" + re.escape(e["name"].lower()) + r"(?![a-z0-9])", text)]
    defaults = {"PDAC": ["Gemcitabine", "5-Fluorouracil", "Erlotinib", "Aspirin"],
                "NSCLC_KRAS": ["Sotorasib", "Erlotinib"], "AMR_BACTERIAL": ["Meropenem", "Ciprofloxacin"],
                "METABOLIC_MASLD": ["Resmetirom", "Metformin"]}
    return {
        "prompt": prompt, "candidate_count": count, "disease_id": disease,
        "priorities": priorities or ["balanced"], "objective_weights": weights,
        "seed_queries": explicit_seeds or defaults[disease],
        "generations": 2, "population_size": max(12, count * 2), "top_k_cellular": 1,
        "provider": provider,
        "explanation": "The search generates RDKit-valid variants and ranks measured model outputs using these priorities. The AI guides the search policy; it does not supply efficacy values. Artificial universe parameters are unchanged.",
        "limitation": "Produces experimental candidates, not cures. Drug matching covers the local reference library only." + (" This disease has no novel-compound activity predictor: ranking uses physicochemical properties, risk alerts and domain similarity only; cellular efficacy is not simulated." if disease != "PDAC" else " Activity uses an exploratory PDAC surrogate, not clinical validation."),
    }


@lru_cache(maxsize=1)
def local_structure_corpus():
    """Every distinct structure EVOCURE holds locally, for prior art checks.

    Combines the curated compound library with the reference bioactivity dataset.
    This is around 120 structures, which is a sanity check and emphatically not a
    prior art search; see the caveat returned by assess_local_novelty.
    """
    from rdkit.Chem.Scaffolds import MurckoScaffold
    import json
    from pathlib import Path

    generator = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
    corpus: dict[str, dict] = {}

    def add(name, smiles, source, category=""):
        mol = Chem.MolFromSmiles(smiles) if smiles else None
        if mol is None:
            return
        canonical = Chem.MolToSmiles(mol, isomericSmiles=True)
        if canonical in corpus:
            return
        try:
            scaffold = MurckoScaffold.MurckoScaffoldSmiles(mol=mol)
        except Exception:
            scaffold = ""
        corpus[canonical] = {
            "name": name, "source": source, "category": category,
            "canonical_smiles": canonical,
            "connectivity": Chem.MolToSmiles(mol, isomericSmiles=False),
            "scaffold": scaffold,
            "inchikey": Chem.MolToInchiKey(mol),
            "fingerprint": generator.GetFingerprint(mol),
        }

    for entry in COMPOUND_LIBRARY:
        add(entry.get("name", "unnamed"), entry.get("smiles"),
            "EVOCURE compound library", entry.get("category", ""))

    dataset = reference_data_dir() / "pdac_reference_dataset.json"
    if dataset.is_file():
        payload = json.loads(dataset.read_text(encoding="utf-8"))
        for entry in payload.get("compounds", []):
            add(entry.get("name", "unnamed"), entry.get("canonical_smiles"),
                "Curated reference dataset", entry.get("disease_id", ""))

    return list(corpus.values())


def assess_local_novelty(compound, top_n: int = 5) -> dict:
    """Compare a structure against everything held locally.

    Answers "does something like this already exist here", which is the most that
    can be established offline. It cannot answer whether the structure is new to
    the world: that needs PubChem, ChEMBL, SureChEMBL and a patent search.
    """
    mol = Chem.MolFromSmiles(compound.canonical_smiles)
    if mol is None:
        raise ValueError("Candidate has no valid molecular structure")

    from rdkit.Chem.Scaffolds import MurckoScaffold

    generator = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
    fingerprint = generator.GetFingerprint(mol)
    canonical = Chem.MolToSmiles(mol, isomericSmiles=True)
    connectivity = Chem.MolToSmiles(mol, isomericSmiles=False)
    inchikey = Chem.MolToInchiKey(mol)
    try:
        scaffold = MurckoScaffold.MurckoScaffoldSmiles(mol=mol)
    except Exception:
        scaffold = ""

    corpus = local_structure_corpus()
    scored = []
    for entry in corpus:
        similarity = float(DataStructs.TanimotoSimilarity(fingerprint, entry["fingerprint"]))
        scored.append({
            "name": entry["name"], "source": entry["source"], "category": entry["category"],
            "canonical_smiles": entry["canonical_smiles"],
            "similarity": round(similarity, 4),
            "exact_match": entry["canonical_smiles"] == canonical,
            "same_connectivity": entry["connectivity"] == connectivity,
            "same_scaffold": bool(scaffold) and entry["scaffold"] == scaffold,
            # First block of an InChIKey is the skeleton; a match means the same
            # constitution ignoring stereochemistry and protonation.
            "same_skeleton": entry["inchikey"][:14] == inchikey[:14],
        })
    scored.sort(key=lambda m: (m["exact_match"], m["similarity"]), reverse=True)

    exact = next((m["name"] for m in scored if m["exact_match"]), None)
    skeleton = next((m["name"] for m in scored if m["same_skeleton"] and not m["exact_match"]), None)
    scaffold_hits = [m["name"] for m in scored if m["same_scaffold"]][:5]
    best = scored[0]["similarity"] if scored else 0.0

    if exact:
        classification = "Already present locally: exact structural match"
    elif skeleton:
        classification = "Same skeleton as a known compound; differs in stereochemistry or charge"
    elif best >= 0.85:
        classification = "Very close analogue of a known compound"
    elif best >= 0.70:
        classification = "Close analogue of a known compound"
    elif best >= 0.40:
        classification = "Related to known chemistry, clearly distinct structure"
    else:
        classification = "No close relative in the local corpus"

    return {
        "corpus_size": len(corpus),
        "corpus_scope": (
            "EVOCURE compound library plus the curated reference dataset. Roughly 120 "
            "structures held locally."
        ),
        "max_similarity": round(best, 4),
        "exact_match": exact,
        "same_skeleton_match": skeleton,
        "same_scaffold_matches": scaffold_hits,
        "classification": classification,
        "nearest": scored[:max(1, min(20, top_n))],
        "method": (
            "RDKit Morgan fingerprint, radius 2, 2048 bits, Tanimoto similarity. Exact "
            "identity uses canonical isomeric SMILES; skeleton identity uses the first "
            "InChIKey block; scaffold identity uses Bemis-Murcko."
        ),
        "caveat": (
            "This searches roughly 120 local structures. It is a sanity check, not a prior "
            "art search. A structure absent here may still be published, catalogued or "
            "patented, and nothing in this result establishes that it is new. Confirming "
            "novelty requires PubChem, ChEMBL, SureChEMBL and a patent search, none of "
            "which run offline. Similarity to an active drug does not transfer that "
            "drug's activity, and distance from one is not evidence of anything."
        ),
    }
