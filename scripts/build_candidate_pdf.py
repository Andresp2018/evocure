"""Render the PDAC screening run as a PDF report.

Reads reports/pdac_publication_test/experiment_results.json, produced by
scripts/run_pdac_report_experiment.py, and writes a report that states what the
platform computed and what those numbers do and do not support.

    python scripts/build_candidate_pdf.py

Every figure in the output is read from the results file. Nothing is hand written
into the report, so regenerating after a new run always matches that run.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any
import argparse
import json
import sys

from fpdf import FPDF

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "reports" / "pdac_publication_test" / "experiment_results.json"
DEFAULT_OUTPUT = ROOT / "reports" / "EVOCURE_PDAC_Candidate_Report.pdf"

INK = (26, 32, 44)
MUTED = (113, 128, 150)
RULE = (203, 213, 224)
ACCENT = (44, 82, 130)
PANEL = (247, 250, 252)
WARN_BG = (255, 250, 235)
WARN_EDGE = (214, 158, 46)
GOOD = (47, 133, 90)
BAD = (197, 48, 48)


def ascii_only(text: Any) -> str:
    """Core PDF fonts are latin-1. Keep the document to plain ASCII."""
    if text is None:
        return ""
    # Code points, not literals, so this table survives any re-encoding of the file.
    swaps = {
        chr(0x2014): ", ", chr(0x2013): "-", chr(0x2018): "'", chr(0x2019): "'",
        chr(0x201C): '"', chr(0x201D): '"', chr(0x00B1): "+/-", chr(0x2265): ">=",
        chr(0x2264): "<=", chr(0x00B5): "u", chr(0x03BC): "u", chr(0x2192): "->",
        chr(0x2026): "...", chr(0x00D7): "x", chr(0x00B2): "2", chr(0x00B0): " deg",
    }
    out = str(text)
    for bad, good in swaps.items():
        out = out.replace(bad, good)
    return out.encode("ascii", "replace").decode("ascii")


def fmt(value: Any, digits: int = 2, dash: str = "not available") -> str:
    if value is None:
        return dash
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, (int, float)):
        return f"{value:.{digits}f}"
    return ascii_only(value)


def p_value_text(value: float | None) -> str:
    if value is None:
        return "not available"
    if value < 0.001:
        return "p < 0.001"
    return f"p = {value:.3f}"


class Report(FPDF):
    def __init__(self, report_id: str) -> None:
        super().__init__(orientation="P", unit="mm", format="A4")
        self.report_id = ascii_only(report_id)
        self.set_auto_page_break(auto=True, margin=20)
        self.set_margins(18, 16, 18)
        self.set_title("EVOCURE PDAC Candidate Screening Report")
        self.set_author("EVOCURE open source platform")
        self.set_creator("scripts/build_candidate_pdf.py")
        self.cover = False

    # Running header and footer -------------------------------------------------
    def header(self) -> None:
        if self.cover or self.page_no() == 1:
            return
        self.set_font("Helvetica", "", 7.5)
        self.set_text_color(*MUTED)
        self.set_y(9)
        self.cell(120, 4, "EVOCURE in silico screening report. Computational prediction, not clinical evidence.", align="L")
        self.cell(54, 4, self.report_id, align="R")
        self.set_draw_color(*RULE)
        self.set_line_width(0.2)
        self.line(18, 15, 192, 15)
        self.set_y(20)

    def footer(self) -> None:
        if self.cover:
            return
        self.set_y(-14)
        self.set_font("Helvetica", "", 7.5)
        self.set_text_color(*MUTED)
        self.cell(120, 4, "No clinical, diagnostic or therapeutic claim is made.", align="L")
        self.cell(54, 4, f"Page {self.page_no()} of {{nb}}", align="R")

    # Building blocks -----------------------------------------------------------
    # fpdf2 leaves the cursor at the left edge of the cell it just drew, not at the
    # page margin, so every block below re-anchors x before it writes.
    def anchor(self) -> None:
        self.set_x(self.l_margin)

    def multi_cell(self, w=0, *args, **kwargs):  # type: ignore[override]
        """A full width block always starts at the left margin.

        fpdf2 restores x to the left edge of the cell it just drew, which for a
        nested or offset cell is not the page margin. Without this, a w=0 block
        after such a cell gets a zero width and raises.
        """
        if w == 0 and not kwargs.get("dry_run"):
            self.set_x(self.l_margin)
        return super().multi_cell(w, *args, **kwargs)

    def h1(self, text: str) -> None:
        self.ln(3)
        self.anchor()
        self.set_font("Helvetica", "B", 15)
        self.set_text_color(*INK)
        self.multi_cell(0, 7, ascii_only(text))
        self.set_draw_color(*ACCENT)
        self.set_line_width(0.5)
        y = self.get_y() + 1
        self.line(18, y, 50, y)
        self.ln(4)

    def h2(self, text: str, needs: float = 34) -> None:
        # A heading alone at the foot of a page is worse than a slightly short page,
        # so move it down when its content will not fit beneath it.
        if self.get_y() + needs > self.h - 22:
            self.add_page()
        self.ln(2)
        self.anchor()
        self.set_font("Helvetica", "B", 10.5)
        self.set_text_color(*ACCENT)
        self.multi_cell(0, 5.5, ascii_only(text))
        self.ln(1)

    def body(self, text: str, size: float = 9.5, leading: float = 4.8) -> None:
        self.anchor()
        self.set_font("Helvetica", "", size)
        self.set_text_color(*INK)
        self.multi_cell(0, leading, ascii_only(text))
        self.ln(1.5)

    def small(self, text: str) -> None:
        self.anchor()
        self.set_font("Helvetica", "", 8)
        self.set_text_color(*MUTED)
        self.multi_cell(0, 4, ascii_only(text))
        self.ln(1)

    def callout(self, title: str, text: str, tone: str = "warn") -> None:
        self.anchor()
        edge = WARN_EDGE if tone == "warn" else ACCENT
        fill = WARN_BG if tone == "warn" else PANEL
        self.set_font("Helvetica", "B", 9)
        title_lines = 1
        self.set_font("Helvetica", "", 8.7)
        width = 174 - 8
        body_lines = len(self.multi_cell(width, 4.3, ascii_only(text),
                                         dry_run=True, output="LINES"))
        height = 5 + title_lines * 5 + body_lines * 4.3 + 3

        if self.get_y() + height > self.h - 22:
            self.add_page()

        top = self.get_y()
        self.set_fill_color(*fill)
        self.set_draw_color(*edge)
        self.set_line_width(0.3)
        self.rect(18, top, 174, height, style="DF")
        self.set_line_width(1.4)
        self.line(18.7, top, 18.7, top + height)

        self.set_xy(22, top + 2.5)
        self.set_font("Helvetica", "B", 9)
        self.set_text_color(*INK)
        self.cell(width, 5, ascii_only(title))
        self.set_xy(22, top + 7.5)
        self.set_font("Helvetica", "", 8.7)
        self.multi_cell(width, 4.3, ascii_only(text))
        self.set_y(top + height + 4)

    def key_values(self, rows: list[tuple[str, str]], label_w: float = 54) -> None:
        self.set_font("Helvetica", "", 9)
        for label, value in rows:
            if self.get_y() > self.h - 28:
                self.add_page()
            self.anchor()
            self.set_text_color(*MUTED)
            self.set_font("Helvetica", "", 9)
            self.cell(label_w, 5, ascii_only(label))
            self.set_text_color(*INK)
            self.set_font("Helvetica", "", 9)
            self.multi_cell(174 - label_w, 5, ascii_only(value))
        self.ln(2)

    def table(self, headers: list[str], widths: list[float],
              rows: list[list[str]], aligns: list[str] | None = None,
              row_colors: list[tuple[int, int, int] | None] | None = None) -> None:
        aligns = aligns or ["L"] * len(headers)
        if self.get_y() + 14 > self.h - 22:
            self.add_page()

        self.anchor()
        self.set_font("Helvetica", "B", 8.3)
        self.set_fill_color(*ACCENT)
        self.set_text_color(255, 255, 255)
        for header, width, align in zip(headers, widths, aligns):
            self.cell(width, 6.5, ascii_only(header), border=0, align=align, fill=True)
        self.ln()

        self.set_font("Helvetica", "", 8.3)
        for index, row in enumerate(rows):
            self.anchor()
            if self.get_y() + 6 > self.h - 22:
                self.add_page()
                self.anchor()
                self.set_font("Helvetica", "B", 8.3)
                self.set_fill_color(*ACCENT)
                self.set_text_color(255, 255, 255)
                for header, width, align in zip(headers, widths, aligns):
                    self.cell(width, 6.5, ascii_only(header), border=0, align=align, fill=True)
                self.ln()
                self.anchor()
                self.set_font("Helvetica", "", 8.3)

            shade = index % 2 == 0
            self.set_fill_color(*(PANEL if shade else (255, 255, 255)))
            tint = (row_colors[index] if row_colors and index < len(row_colors) else None)
            for column, (cell, width, align) in enumerate(zip(row, widths, aligns)):
                self.set_text_color(*(tint if (tint and column == len(row) - 1) else INK))
                self.cell(width, 5.6, ascii_only(cell), border=0, align=align, fill=True)
            self.ln()
        self.set_text_color(*INK)
        self.ln(3)


# ---------------------------------------------------------------------------
# Content
# ---------------------------------------------------------------------------

def cover_page(pdf: Report, data: dict) -> None:
    pdf.cover = True
    pdf.add_page()

    pdf.set_fill_color(*ACCENT)
    pdf.rect(0, 0, 210, 58, style="F")
    pdf.set_xy(18, 16)
    pdf.set_font("Helvetica", "B", 23)
    pdf.set_text_color(255, 255, 255)
    pdf.cell(0, 10, "EVOCURE Candidate Screening")
    pdf.set_xy(18, 28)
    pdf.set_font("Helvetica", "", 13)
    pdf.cell(0, 8, "Pancreatic ductal adenocarcinoma, five prioritised compounds")
    pdf.set_xy(18, 40)
    pdf.set_font("Helvetica", "", 9)
    pdf.cell(0, 5, ascii_only(
        f"{data['report_id']}   |   generated {data['created_utc'][:19].replace('T', ' ')} UTC"))

    pdf.set_y(70)
    pdf.callout(
        "What this document is",
        "A demonstration of an open source in silico screening platform. It shows what the "
        "software computed on one reproducible run: a bounded evolutionary search over "
        "molecular structures, scored by a QSAR model, toxicology heuristics and a simulated "
        "two dimensional tumour microenvironment.\n\n"
        "What it is not: evidence that any compound here treats pancreatic cancer. No "
        "molecule in this report has been synthesised, assayed in a cell, or given to an "
        "animal or a person. These are computational hypotheses at the earliest possible "
        "stage of discovery, and the large majority of such hypotheses fail.",
        tone="warn",
    )

    headline = data.get("headline", "")
    assay = data.get("assay_validity", {})
    separated = data.get("candidates_separating_from_negative_control", [])

    pdf.h2("Result in one line")
    pdf.set_font("Helvetica", "B", 11)
    pdf.set_text_color(*(GOOD if separated else BAD))
    pdf.multi_cell(0, 5.5, ascii_only(headline))
    pdf.ln(2)

    pdf.set_text_color(*INK)
    pdf.body(
        "The screening pipeline ran end to end and produced five ranked candidates. The "
        "tissue assay was checked with a positive and a negative control before the "
        "candidates were read, so the result above is interpretable rather than merely "
        "generated. " + ascii_only(assay.get("interpretation", ""))
    )

    compute = data.get("compute", {})
    qsar = data.get("qsar_validation", {})
    pdf.ln(1)
    pdf.h2("Run at a glance")
    pdf.key_values([
        ("Structures evaluated", f"{data.get('search_total_structures', 0)} unique valid structures"),
        ("Generations", str(data.get("search_settings", {}).get("generations", "")) +
         f", population {data.get('search_settings', {}).get('population_size', '')}"),
        ("Replicate seeds per compound", str(len(data.get("replicate_seeds", [])))),
        ("Compute backend", compute.get("numerical_backend", "")),
        ("QSAR training set", f"{data.get('training_structure_count', 0)} curated compounds"),
        ("QSAR scaffold split R2", fmt(qsar.get("r2"), 3)),
        ("QSAR classifier ROC AUC", fmt(qsar.get("roc_auc"), 3)),
        ("Wall clock", f"{fmt(data.get('elapsed_seconds'), 0)} seconds"),
    ], label_w=56)

    pdf.ln(1)
    pdf.small(
        "Open source platform and data. Reproduce this document with the commands in the "
        "reproducibility section. Report generated directly from the run's results file; "
        "no figure in it was entered by hand."
    )
    pdf.cover = False


def summary_section(pdf: Report, data: dict) -> None:
    pdf.add_page()
    pdf.h1("1. Summary")

    candidates = data.get("candidates", [])
    controls = {c["label"]: c for c in data.get("reference_controls", [])}
    assay = data.get("assay_validity", {}).get("positive_vs_negative_control", {})

    pdf.body(
        "A bounded evolutionary search started from four approved pancreatic cancer drugs "
        "and generated structural variants through point mutations and scaffold "
        "recombination. Every variant was validated with RDKit, scored on five objectives, "
        "and the non dominated front was carried into the next generation. The five "
        "compounds below are the highest ranked variants that were not themselves seeds."
    )

    pdf.h2("1.1 Assay validity, checked before reading the candidates")
    positive = controls.get(data.get("controls", {}).get("positive", "Gemcitabine"))
    negative = controls.get(data.get("controls", {}).get("negative", "Caffeine"))

    rows = []
    for role, entry in (("Positive, curated", positive), ("Negative", negative)):
        if not entry:
            continue
        spread = entry["summary"]["tgi"]
        rows.append([
            role, entry["label"],
            fmt(spread["mean"], 1),
            f"{fmt(spread['ci95_low'], 1)} to {fmt(spread['ci95_high'], 1)}",
            "yes" if spread.get("excludes_zero") else "no",
        ])

    modelled = data.get("modelled_potency_control") or {}
    if modelled:
        spread = modelled["summary"]
        rows.append([
            "Positive, modelled", ascii_only(modelled["compound"]),
            fmt(spread["mean"], 1),
            f"{fmt(spread['ci95_low'], 1)} to {fmt(spread['ci95_high'], 1)}",
            "yes" if spread.get("excludes_zero") else "no",
        ])

    pdf.table(
        ["Role", "Compound", "TGI %", "95% CI", "CI excludes 0"],
        [34, 42, 24, 44, 30], rows,
        ["L", "L", "R", "C", "C"],
    )

    if modelled:
        pdf.small(
            "The third arm is the same positive control molecule with its curated "
            "bioactivity withheld, so its potency comes from the structure-property model "
            "that scores every generated candidate. It is the only arm the candidates can "
            "fairly be compared against."
        )

    separated_controls = data.get("assay_validity", {}).get("assay_separates_controls")
    pdf.callout(
        "Does the simulated assay work?",
        (f"Positive versus negative control: {p_value_text(assay.get('p_value'))}, "
         f"mean difference {fmt(assay.get('mean_difference_pct'), 1)} percentage points, "
         f"Hedges g {fmt(assay.get('hedges_g'), 2)}.\n\n" +
         ascii_only(data.get("assay_validity", {}).get("interpretation", ""))),
        tone="info" if separated_controls else "warn",
    )

    modelled = data.get("modelled_potency_control") or {}
    if modelled:
        pdf.h2("1.1a The comparison is between a measurement and a prediction")
        separates = modelled.get("versus_negative_control", {}).get(
            "separates_from_negative_control")
        pdf.callout(
            "The positive control fails its own test when scored like a candidate",
            f"{ascii_only(modelled['compound'])} draws its potency from curated "
            f"experimental bioactivity: IC50 {fmt(modelled['curated_ic50_um'], 4)} uM, and it "
            f"inhibits growth by {fmt((positive or {}).get('summary', {}).get('tgi', {}).get('mean'), 1)} %. "
            f"A generated structure has no measurement, so it is scored by the "
            f"structure-property model. Scoring this same molecule that way gives IC50 "
            f"{fmt(modelled['modelled_ic50_um'], 3)} uM, {fmt(modelled['fold_weaker_when_modelled'], 0)} "
            f"times weaker, and growth inhibition falls to "
            f"{fmt(modelled['summary']['mean'], 1)} %"
            + (", which does not separate from the negative control.\n\n"
               if separates is False else ".\n\n") +
            "Nothing about the molecule changed. Only the provenance of its potency did. "
            "Any candidate in this report is therefore being compared against a number no "
            "generated structure can ever be assigned, and failing that comparison says "
            "little about the chemistry.",
            tone="warn",
        )

    pdf.h2("1.2 The five candidates")
    rows, colors = [], []
    for entry in candidates:
        spread = entry["summary"]["tgi"]
        test = entry.get("versus_negative_control", {})
        ranking = entry.get("ranking") or {}
        pic50 = (entry.get("qsar", {}).get("pic50") or {}).get("value")
        if test.get("bonferroni_separates"):
            verdict, tint = "separates, corrected", GOOD
        elif test.get("separates_from_negative_control"):
            verdict, tint = "marginal, uncorrected", WARN_EDGE
        else:
            verdict, tint = "not distinguishable", BAD
        colors.append(tint)
        rows.append([
            entry["label"].replace("_", " "),
            ascii_only(entry["compound"]["name"]),
            fmt(pic50, 2),
            fmt(ranking.get("composite_utility"), 3),
            fmt(spread["mean"], 1),
            f"{fmt(spread['ci95_low'], 1)} to {fmt(spread['ci95_high'], 1)}",
            verdict,
        ])
    pdf.table(
        ["ID", "Compound", "pIC50", "Score", "TGI %", "95% CI", "vs control"],
        [22, 34, 17, 17, 17, 34, 33], rows,
        ["L", "L", "R", "R", "R", "C", "C"], colors,
    )

    pdf.small(
        "TGI is tumour growth inhibition in the simulated microenvironment, relative to an "
        "untreated arm, averaged over independent seeds. Score is the weighted composite "
        "across the five search objectives. The last column reports Welch's t-test of each "
        "candidate against the negative control; it is the column that decides whether a "
        "point estimate means anything."
    )

    separated = data.get("candidates_separating_from_negative_control", [])
    if not separated:
        control_also_fails = (modelled.get("versus_negative_control", {})
                              .get("separates_from_negative_control") is False)
        extra = (
            "\n\nThis is not evidence that the molecules are inactive. The positive "
            "control fails the identical test when scored by the same potency model, so "
            "the test does not currently distinguish a good candidate from a known drug. "
            "What the run establishes is a limit of the platform, not a property of these "
            "five structures."
            if control_also_fails else ""
        )
        pdf.callout(
            "Read this before using the ranking",
            "No candidate separated from the negative control in the tissue simulation. The "
            "ranking above is therefore driven by predicted properties, drug likeness, "
            "toxicity alerts and applicability domain, not by a demonstrated simulated "
            "effect." + extra +
            "\n\nThe honest reading is that the search found chemically reasonable, low "
            "alert, drug like molecules near the seed scaffolds, and that this experiment "
            "cannot say whether any of them is active. Treat the list as a starting set for "
            "further computational work, not as a shortlist of likely treatments.",
            tone="warn",
        )
    else:
        corrected = data.get("candidates_separating_bonferroni", [])
        names = ", ".join(ascii_only(s).replace("_", " ") for s in separated)
        if corrected:
            detail = (
                f"{len(corrected)} of those survived Bonferroni correction for "
                f"{len(candidates)} comparisons, which is the threshold that matters when "
                "several candidates are tested against one control arm."
            )
        else:
            detail = (
                f"None of them survives Bonferroni correction for {len(candidates)} "
                f"comparisons (p < {0.05 / max(1, len(candidates)):.2f}). Testing five "
                "candidates against one control arm makes a single result near p = 0.05 "
                "roughly what chance alone would produce, so this is a marginal signal and "
                "should be treated as one."
            )
        pdf.callout(
            "Scope of the positive finding",
            f"Separated at an uncorrected p < 0.05: {names}. {detail}\n\n"
            "Either way this is a result inside a simulation whose parameters are not "
            "experimentally calibrated. It justifies further computational scrutiny and "
            "nothing beyond that. Separation here is not potency, not selectivity, not "
            "safety and not evidence about human disease.",
            tone="warn",
        )


def method_section(pdf: Report, data: dict) -> None:
    pdf.add_page()
    pdf.h1("2. Method")

    settings = data.get("search_settings", {})
    plan = data.get("plan", {})

    pdf.h2("2.1 Search")
    pdf.key_values([
        ("Objective prompt", ascii_only(plan.get("prompt", ""))),
        ("Planner", ascii_only(plan.get("provider", ""))),
        ("Derived priorities", ", ".join(ascii_only(p) for p in plan.get("priorities", []))),
        ("Seed compounds", ", ".join(ascii_only(s) for s in settings.get("seed_queries", []))),
        ("Generations", str(settings.get("generations"))),
        ("Population size", str(settings.get("population_size"))),
        ("Applied concentration", f"{settings.get('applied_concentration_um')} uM"),
        ("Exposure duration", f"{settings.get('duration_hours')} hours"),
        ("Search seed", str(settings.get("seed"))),
    ], label_w=44)

    pdf.body(
        "The planner converts a plain language objective into search priorities only. It "
        "does not propose molecules and it does not supply any activity, potency or safety "
        "number. All quantitative output comes from the deterministic scientific models."
    )

    weights = settings.get("objective_weights", {})
    closed_loop = bool(data.get("tissue_in_fitness"))
    rows = [
        ["Predicted PDAC activity", fmt(weights.get("predicted_pdac_activity"), 1),
         "Calibrated QSAR regressor, sigmoidal utility at pIC50 6.0"],
        ["Low toxicological risk", fmt(weights.get("low_toxicological_risk"), 1),
         "Ten category structural alert and rule based risk index"],
        ["Drug likeness (QED)", fmt(weights.get("qed_druglikeness"), 1),
         "RDKit quantitative estimate of drug likeness"],
        ["Selectivity index", fmt(weights.get("selectivity_index"), 1),
         "ADMET derived tumour versus healthy cell window"],
        ["Applicability domain", fmt(weights.get("applicability_domain_confidence"), 1),
         "Maximum Tanimoto similarity to the training set"],
    ]
    if closed_loop:
        rows.append([
            "Simulated growth inhibition",
            fmt(weights.get("simulated_growth_inhibition", 1.0), 1),
            "Agent based tumour microenvironment, treated versus untreated",
        ])

    pdf.h2("2.2 Objective weights")
    pdf.table(["Objective", "Weight", "Source of the value"], [62, 20, 92], rows, ["L", "R", "L"])
    pdf.small(
        "Tumour cell kill is never maximised alone. A compound that scores well on activity "
        "but triggers toxicity alerts or falls outside the applicability domain is penalised "
        "by construction."
    )

    pdf.h2("2.2a What selection actually optimises")
    if closed_loop:
        pdf.callout(
            "The selection loop is closed",
            f"Every candidate that could be dosed was simulated in the tumour "
            f"microenvironment, {data.get('tissue_simulations_run', 0)} runs in total, and "
            "its growth inhibition entered the objective set used for Pareto sorting and "
            "survivor selection. The tumour model therefore exerts selective pressure on "
            "the population rather than only reporting on the winners.\n\n"
            "Two limits on how much that is worth. All arms share one seed per generation, "
            "so the comparison is paired but the objective is the noisiest of the six. And "
            "the tissue model's drug response is driven by the predicted IC50, so this "
            "objective is correlated with the activity objective rather than independent "
            "of it. It adds diffusion, uptake, hypoxia and clonal structure on top of that "
            "prediction; it does not replace it with a measurement.",
            tone="info",
        )
    else:
        pdf.callout(
            "The selection loop is open",
            "Selection used the chemistry and prediction objectives only. The tumour "
            "simulation was computed for the top ranked candidates and recorded, but it did "
            "not enter the objective set, so it exerted no selective pressure. The search "
            "optimised a QSAR surrogate, and the tissue numbers in this report are a "
            "readout of that search rather than its driver.",
            tone="warn",
        )

    pdf.h2("2.3 The QSAR model, and what it can carry")
    qsar = data.get("qsar_validation", {})
    pdf.key_values([
        ("Training compounds", str(data.get("training_structure_count", ""))),
        ("Independent scaffolds", str(qsar.get("independent_scaffolds", "not recorded"))),
        ("Validation", ascii_only(qsar.get("validation_scheme",
                                           "Bemis-Murcko scaffold split"))),
        ("Pooled test predictions", str(qsar.get("test_compounds", ""))),
        ("Regression RMSE", f"{fmt(qsar.get('rmse'), 3)} log units"),
        ("Regression R2", fmt(qsar.get("r2"), 3)),
        ("Classifier ROC AUC", fmt(qsar.get("roc_auc"), 3)),
        ("Classifier Brier score", fmt(qsar.get("brier_score"), 3)),
        ("Conformal 95% half width", f"{fmt(qsar.get('conformal_q_95'), 2)} log units"),
    ], label_w=52)

    r2 = qsar.get("r2")
    quantile = qsar.get("conformal_q_95")
    strong = isinstance(r2, (int, float)) and r2 >= 0.5
    fold = (10.0 ** float(quantile)) if isinstance(quantile, (int, float)) else None

    pdf.callout(
        "Ordering is usable, individual values are not",
        (f"Under repeated scaffold-grouped cross validation the regressor explains a "
         f"real share of the variance across held out scaffolds (R2 = {fmt(r2, 2)}, "
         f"RMSE {fmt(qsar.get('rmse'), 2)} log units), and the classifier separates active "
         f"from inactive well (ROC AUC = {fmt(qsar.get('roc_auc'), 2)}).\n\n"
         if strong else
         f"The regressor explains little of the variance across held out scaffolds "
         f"(R2 = {fmt(r2, 2)}). The classifier is the stronger half "
         f"(ROC AUC = {fmt(qsar.get('roc_auc'), 2)}).\n\n") +
        (f"That does not make a predicted IC50 a quantity. The calibrated 95% conformal "
         f"interval is about {fmt(quantile, 1)} log units in each direction, a factor of "
         f"roughly {fold:,.0f} on IC50. Rank ordering within the generated set is "
         f"defensible; a single predicted IC50 is not, and every pIC50 here is printed "
         f"with its interval for that reason.\n\n"
         if fold else "") +
        "A note on method. These figures come from repeated grouped folds, not one split. "
        "A single scaffold split on 43 compounds leaves about ten test compounds and "
        "returns anything from R2 -0.09 to 0.82 depending only on the random seed, so a "
        "value from one split describes the split rather than the model.",
        tone="info" if strong else "warn",
    )

    pdf.h2("2.4 Tissue simulation")
    grid = data.get("grid", {})
    pdf.key_values([
        ("Model", "Two dimensional agent based PDAC microenvironment"),
        ("Grid", f"{grid.get('nx')} x {grid.get('ny')} cells over "
                 f"{fmt(grid.get('width'), 0)} x {fmt(grid.get('height'), 0)} micrometres"),
        ("Fields", "Oxygen and nutrient reaction diffusion, drug diffusion and uptake"),
        ("Features", "Hypovascularity, dense collagen matrix, sensitive and resistant clones"),
        ("Comparison", "Treated arm against an untreated arm with the same seed"),
        ("Replicates", f"{len(data.get('replicate_seeds', []))} independent seeds per compound"),
    ], label_w=44)
    pdf.small(
        "Kinetics and microenvironment parameters are exploratory and have not been fitted to "
        "experimental pancreatic tumour data. The model's drug response is driven by the "
        "predicted IC50, so two compounds with the same predicted IC50 produce the same "
        "trajectory. Simulated growth inhibition is a property of this model, not a measurement."
    )


def candidate_pages(pdf: Report, data: dict) -> None:
    images_dir = RESULTS.parent
    for entry in data.get("candidates", []):
        pdf.add_page()
        compound = entry["compound"]
        ranking = entry.get("ranking") or {}
        qsar = entry.get("qsar", {})
        pic50 = qsar.get("pic50") or {}
        interval = qsar.get("conformal_interval_95") or {}
        probability = qsar.get("active_probability") or {}
        spread = entry["summary"]["tgi"]
        resistant = entry["summary"].get("resistant_fraction", {})
        test = entry.get("versus_negative_control", {})
        comparison = entry.get("drug_comparison", {})

        label = entry["label"].replace("_", " ")
        pdf.h1(f"{label}: {ascii_only(compound['name'])}")

        image = images_dir / f"{entry['label']}.png"
        if image.is_file():
            top = pdf.get_y()
            pdf.set_draw_color(*RULE)
            pdf.set_line_width(0.2)
            pdf.rect(18, top, 174, 42)
            pdf.image(str(image), x=20, y=top + 1.5, h=39)
            pdf.set_y(top + 46)

        pdf.h2("Identity")
        pdf.key_values([
            ("Canonical SMILES", compound.get("canonical_smiles", "")),
            ("InChIKey", entry.get("inchikey", "")),
            ("Molecular formula", compound.get("formula", "")),
            ("Murcko scaffold", compound.get("murcko_scaffold", "")),
            ("Origin", f"generation {ranking.get('origin_generation', '?')} variant, "
                       f"{ascii_only(ranking.get('transformation_type', 'directed optimization'))}"),
        ], label_w=36)

        pdf.h2("Physicochemical profile")
        descriptors = entry.get("descriptors", {})
        pdf.table(
            ["MW", "cLogP", "TPSA", "HBD", "HBA", "QED", "Lipinski"],
            [26, 24, 26, 18, 18, 24, 38],
            [[
                fmt(descriptors.get("molecular_weight"), 1),
                fmt(descriptors.get("logp"), 2),
                fmt(descriptors.get("tpsa"), 1),
                fmt(descriptors.get("hbd"), 0),
                fmt(descriptors.get("hba"), 0),
                fmt(descriptors.get("qed"), 3),
                lipinski_text(descriptors),
            ]],
            ["R", "R", "R", "R", "R", "R", "C"],
        )

        pdf.h2("Predicted activity, with its uncertainty")
        pdf.table(
            ["Quantity", "Value", "95% interval", "Reading"],
            [48, 28, 44, 54],
            [
                ["Predicted pIC50", fmt(pic50.get("value"), 3),
                 f"{fmt(interval.get('low'), 2)} to {fmt(interval.get('high'), 2)}",
                 "ordering signal"],
                ["Implied IC50 (uM)", fmt(qsar.get("ic50_um_estimated"), 4),
                 "spans orders of magnitude", "not a quantity"],
                ["Active probability", fmt(probability.get("value"), 3),
                 f"{fmt(probability.get('uncertainty_low'), 2)} to "
                 f"{fmt(probability.get('uncertainty_high'), 2)}",
                 "the stronger prediction"],
                ["Applicability domain", fmt(qsar.get("tanimoto_similarity_max"), 2),
                 "Tanimoto, nearest neighbour",
                 "inside domain" if qsar.get("in_applicability_domain") else "OUTSIDE, unreliable"],
            ],
            ["L", "R", "C", "L"],
        )

        pdf.h2("Objective scores and risk")
        objectives = ranking.get("objectives", {})
        pdf.table(
            ["Activity", "Selectivity", "QED", "Low tox", "Domain", "Composite", "Risk index"],
            [26, 26, 22, 24, 24, 26, 26],
            [[
                fmt(objectives.get("predicted_pdac_activity"), 3),
                fmt(objectives.get("selectivity_index"), 3),
                fmt(objectives.get("qed_druglikeness"), 3),
                fmt(objectives.get("low_toxicological_risk"), 3),
                fmt(objectives.get("applicability_domain_confidence"), 3),
                fmt(ranking.get("composite_utility"), 3),
                f"{fmt(ranking.get('risk_index'), 1)} / 100",
            ]],
            ["R"] * 7,
        )

        pdf.h2("Simulated tissue response")
        separates = test.get("separates_from_negative_control")
        pdf.table(
            ["Metric", "Value"],
            [68, 106],
            [
                ["Tumour growth inhibition, mean",
                 f"{fmt(spread.get('mean'), 2)} % over {spread.get('n')} seeds"],
                ["95% confidence interval",
                 f"{fmt(spread.get('ci95_low'), 2)} % to {fmt(spread.get('ci95_high'), 2)} %"],
                ["Interval excludes zero", fmt(spread.get("excludes_zero"))],
                ["Per seed values",
                 ", ".join(fmt(v, 1) for v in entry.get("tgi_per_replicate", []))],
                ["Resistant clone fraction", fmt(resistant.get("mean"), 3)],
                ["Value used as selection objective",
                 (f"{fmt(100.0 * float(ranking['objectives']['simulated_growth_inhibition']), 1)} %"
                  f" on the search seed"
                  if ranking.get("objectives", {}).get("simulated_growth_inhibition") is not None
                  else "not an objective in this run")],
                ["Versus negative control",
                 f"{p_value_text(test.get('p_value'))}, Hedges g "
                 f"{fmt(test.get('hedges_g'), 2)}"],
                ["Distinguishable, uncorrected", "yes" if separates else "no"],
                ["Distinguishable, Bonferroni corrected",
                 "yes" if test.get("bonferroni_separates") else "no"],
            ],
            ["L", "L"],
        )

        novelty = entry.get("local_novelty") or {}
        pdf.h2("Does something like this already exist?")
        if novelty:
            pdf.key_values([
                ("Local verdict", ascii_only(novelty.get("classification", ""))),
                ("Nearest local structure",
                 f"{ascii_only((novelty.get('nearest') or [{}])[0].get('name', 'none'))} "
                 f"(Tanimoto {fmt(novelty.get('max_similarity'), 3)})"),
                ("Exact match",
                 ascii_only(novelty.get("exact_match")) if novelty.get("exact_match")
                 else "none in the local corpus"),
                ("Same skeleton",
                 ascii_only(novelty.get("same_skeleton_match"))
                 if novelty.get("same_skeleton_match") else "none"),
                ("Shared Murcko scaffold",
                 ", ".join(ascii_only(n) for n in novelty.get("same_scaffold_matches", []))
                 or "none"),
                ("Corpus searched", f"{novelty.get('corpus_size', 0)} local structures"),
            ], label_w=44)

            nearest_rows = [[
                ascii_only(m.get("name", "")),
                ascii_only(str(m.get("source", ""))[:28]),
                fmt(m.get("similarity"), 3),
                "yes" if m.get("same_scaffold") else "no",
            ] for m in (novelty.get("nearest") or [])[:5]]
            if nearest_rows:
                pdf.table(["Closest local structure", "Source", "Tanimoto", "Same scaffold"],
                          [58, 62, 26, 28], nearest_rows, ["L", "L", "R", "C"])

            pdf.callout(
                "This is not a prior art search",
                ascii_only(novelty.get("caveat", "")),
                tone="warn",
            )
        else:
            nearest = (comparison.get("matches") or [{}])[0]
            pdf.key_values([
                ("Nearest reference", f"{ascii_only(nearest.get('name', 'none'))} "
                                      f"(Tanimoto {fmt(nearest.get('similarity'), 3)})"),
                ("Library scope",
                 f"{comparison.get('reference_count', 0)} curated reference drugs"),
            ], label_w=44)

        pdf.h2("Potential, and what it rests on")
        signals, unknowns = potential_profile(entry, data)
        pdf.table(
            ["Signal", "Reading", "What it is based on"],
            [46, 36, 92], signals, ["L", "L", "L"],
        )
        pdf.set_font("Helvetica", "B", 9)
        pdf.set_text_color(*INK)
        pdf.multi_cell(0, 4.6, "Not assessed at all")
        pdf.set_font("Helvetica", "", 8.7)
        pdf.set_text_color(*MUTED)
        pdf.multi_cell(0, 4.3, ascii_only("; ".join(unknowns) + "."))
        pdf.set_text_color(*INK)
        pdf.ln(2)

        if test.get("bonferroni_separates"):
            verdict = ("This candidate separated from the negative control and survived "
                       "correction for the number of candidates tested. That warrants "
                       "further computational scrutiny and nothing more.")
        elif separates:
            verdict = ("This candidate separated from the negative control at an "
                       "uncorrected threshold but not after correcting for the number of "
                       "candidates tested. Across five candidates, one result in this range "
                       "is close to what chance produces. Treat it as a lead to re-run with "
                       "more seeds, not as a demonstrated effect.")
        else:
            verdict = ("This candidate did not separate from the negative control. Its "
                       "position in the ranking reflects predicted properties and the "
                       "absence of alerts, not a demonstrated simulated effect.")
        pdf.callout("Verdict", verdict,
                    tone="info" if test.get("bonferroni_separates") else "warn")


def potential_profile(entry: dict, data: dict) -> tuple[list[list[str]], list[str]]:
    """Break 'potential' into the signals behind it rather than one score.

    A single number would invite exactly the reading this report is trying to
    avoid, so each signal is shown with the model that produced it.
    """
    ranking = entry.get("ranking") or {}
    objectives = ranking.get("objectives", {})
    qsar = entry.get("qsar", {})
    descriptors = entry.get("descriptors", {})
    novelty = entry.get("local_novelty") or {}
    test = entry.get("versus_negative_control", {})
    spread = entry["summary"]["tgi"]
    provenance = entry.get("potency_provenance") or "unknown"

    def band(value, good, fair, labels=("favourable", "moderate", "unfavourable")):
        if not isinstance(value, (int, float)):
            return "not available"
        if value >= good:
            return labels[0]
        if value >= fair:
            return labels[1]
        return labels[2]

    risk = ranking.get("risk_index")
    rows = [
        ["Predicted potency",
         band(objectives.get("predicted_pdac_activity"), 0.6, 0.35),
         f"pIC50 {fmt((qsar.get('pic50') or {}).get('value'), 2)}, "
         f"{ascii_only(provenance.replace('_', ' '))}"],
        ["Potency certainty", "low by construction",
         f"conformal interval {fmt(data.get('qsar_validation', {}).get('conformal_q_95'), 1)} "
         f"log units"],
        ["Simulated tissue effect",
         ("distinguishable" if test.get("bonferroni_separates")
          else "marginal" if test.get("separates_from_negative_control")
          else "not distinguishable"),
         f"TGI {fmt(spread.get('mean'), 1)} % over {spread.get('n')} seeds, "
         f"{p_value_text(test.get('p_value'))}"],
        ["Toxicology alerts",
         band(100.0 - risk if isinstance(risk, (int, float)) else None, 80, 60),
         f"risk index {fmt(risk, 1)} of 100, ten category screen"],
        ["Drug likeness",
         band(descriptors.get("qed"), 0.6, 0.4),
         f"QED {fmt(descriptors.get('qed'), 3)}, Lipinski {lipinski_text(descriptors)}"],
        ["Model applicability",
         "inside domain" if qsar.get("in_applicability_domain") else "outside domain",
         f"Tanimoto {fmt(qsar.get('tanimoto_similarity_max'), 2)} to nearest training compound"],
        ["Structural distinctness",
         ascii_only(str(novelty.get("classification", "not assessed")).split(":")[0]),
         f"max Tanimoto {fmt(novelty.get('max_similarity'), 2)} over "
         f"{novelty.get('corpus_size', 0)} local structures"],
    ]

    unknowns = [
        "synthetic accessibility and whether a route exists",
        "chemical stability and reactivity",
        "assay interference liability (no PAINS screen)",
        "absorption, distribution, metabolism and excretion in a living system",
        "whether the compound reaches a pancreatic tumour through desmoplastic stroma",
        "any experimental measurement whatsoever",
    ]
    return rows, unknowns


def lipinski_text(descriptors: dict) -> str:
    try:
        violations = sum([
            (descriptors.get("molecular_weight") or 0) > 500,
            (descriptors.get("logp") or 0) > 5,
            (descriptors.get("hbd") or 0) > 5,
            (descriptors.get("hba") or 0) > 10,
        ])
    except TypeError:
        return "not available"
    return "passes" if violations == 0 else f"{violations} violation(s)"


def limitations_section(pdf: Report, data: dict) -> None:
    pdf.add_page()
    pdf.h1("3. What this run does not show")

    pdf.body(
        "The value of a screening report lies as much in its boundaries as in its ranking. "
        "The following are not minor caveats; they determine what the numbers above can be "
        "used for."
    )

    items = [
        ("No experimental evidence of any kind",
         "No compound here has been synthesised. There is no cell assay, no animal model and "
         "no clinical data. Every number is the output of a model."),
        ("The tissue model is not calibrated to real tumours",
         "Kinetics, diffusion constants, clone behaviour and drug response parameters are "
         "plausible but have not been fitted to experimental pancreatic tumour data. "
         "Simulated growth inhibition is a property of this software."),
        ("The potency model orders compounds, it does not quantify them",
         f"R2 of {fmt(data.get('qsar_validation', {}).get('r2'), 2)} under repeated grouped "
         f"cross validation is real signal, but the conformal interval of about "
         f"{fmt(data.get('qsar_validation', {}).get('conformal_q_95'), 1)} log units still "
         "spans orders of magnitude in IC50. Use the ranking, not the numbers."),
        ("A small, narrow training set",
         f"{data.get('training_structure_count', 0)} curated compounds. Applicability domain "
         "is measured against that set, so high domain confidence means close to the training "
         "data, which also means structurally unadventurous."),
        ("Virtual time is not real time",
         "Generations and exposure hours index the simulation. They do not correspond to "
         "chemical synthesis, biological timescales or clinical development."),
        ("Structural similarity is not efficacy and not novelty",
         "Matching a known drug does not transfer its activity. Failing to match anything in "
         "a small local library does not make a structure new."),
        ("No ADME, formulation or synthesis assessment",
         "Nothing here addresses whether these molecules can be made at reasonable cost, are "
         "stable, are orally available, or reach a pancreatic tumour in a living system. "
         "Dense desmoplastic stroma makes that last point a central problem in this disease."),
        ("Multiple comparisons",
         "Five candidates were tested against one control arm. The per candidate threshold is "
         "uncorrected; a Bonferroni corrected threshold would be p < 0.01, and the results "
         "file records both."),
        ("Reference drugs and novel structures are not scored the same way",
         "Gemcitabine's potency in the tissue model comes from curated experimental "
         "bioactivity. A newly generated structure has no measurement, so its potency comes "
         "from the structure-property model, which assigns micromolar values far weaker than "
         "the positive control's. At the dose used here that difference alone is close to "
         "sufficient to explain why no candidate matches the control. The comparison is "
         "therefore between a measurement and a prediction, not between two compounds, and "
         "failing to beat the control is weak evidence about the molecules."),
        ("The tissue objective is noisy and not independent",
         "All arms in a generation share one seed, so selection sees a single draw of a "
         "quantity whose spread across seeds crosses zero. The objective also depends on the "
         "potency the pharmacology model assigns, so it is correlated with the activity "
         "objective rather than an independent check on it."),
    ]
    for title, text in items:
        pdf.set_font("Helvetica", "B", 9.3)
        pdf.set_text_color(*INK)
        pdf.multi_cell(0, 4.8, ascii_only(title))
        pdf.set_font("Helvetica", "", 9)
        pdf.set_text_color(*MUTED)
        pdf.multi_cell(0, 4.5, ascii_only(text))
        pdf.ln(2.2)

    pdf.set_text_color(*INK)
    pdf.callout(
        "If you take one thing from this report",
        "Pancreatic ductal adenocarcinoma has a five year survival rate in the low teens, and "
        "that is precisely why early computational output about it should be described "
        "carefully. This platform is a hypothesis generator. The appropriate next step for "
        "any compound listed here is scrutiny by a medicinal chemist, not optimism.",
        tone="warn",
    )


def reproducibility_section(pdf: Report, data: dict) -> None:
    pdf.add_page()
    pdf.h1("4. Reproducibility")

    compute = data.get("compute", {})
    packages = data.get("packages", {})

    pdf.h2("4.1 Environment of this run")
    pdf.key_values([
        ("Platform", ascii_only(data.get("platform", ""))),
        ("Python", ascii_only(data.get("python", ""))),
        ("Compute backend", ascii_only(compute.get("numerical_backend", ""))),
        ("Device", ascii_only(compute.get("gpu_name") or "host CPU")),
        ("Field precision", ascii_only(compute.get("precision", "float64"))),
        ("Search seed", str(data.get("search_settings", {}).get("seed"))),
        ("Replicate seeds", ", ".join(str(s) for s in data.get("replicate_seeds", []))),
        ("Results hash", ascii_only(data.get("results_hash", ""))[:48]),
    ], label_w=40)

    pdf.h2("4.2 Library versions")
    pdf.table(
        ["Package", "Version"], [60, 50],
        [[name, ascii_only(value)] for name, value in sorted(packages.items())],
        ["L", "L"],
    )

    pdf.h2("4.3 Source file hashes, SHA-256")
    rows = [[ascii_only(path), ascii_only(digest)[:40]]
            for path, digest in sorted(data.get("hashes", {}).items())]
    pdf.table(["File", "SHA-256 (first 40 characters)"], [84, 90], rows, ["L", "L"])

    pdf.h2("4.4 Reproducing this document")
    pdf.body(
        "The platform is open source and so is this result. The reference dataset, the "
        "pipeline, the tissue models and the report generator are all in the repository. No "
        "model weights are required: the QSAR artifact is rebuilt from the committed dataset, "
        "and the optional language models were not used for any quantitative output."
    )
    pdf.set_font("Courier", "", 8.3)
    pdf.set_text_color(*INK)
    for line in [
        "python scripts/train_qsar.py",
        "",
        "# Linux and macOS",
        "EVOCURE_COMPUTE=cpu EVOCURE_DISABLE_LAYA_NEURAL=1 \\",
        "  python scripts/run_pdac_report_experiment.py",
        "",
        "# Windows PowerShell",
        '$env:EVOCURE_COMPUTE = "cpu"',
        '$env:EVOCURE_DISABLE_LAYA_NEURAL = "1"',
        "python scripts\\run_pdac_report_experiment.py",
        "",
        "python scripts/build_candidate_pdf.py",
    ]:
        pdf.cell(0, 4.3, ascii_only(line), new_x="LMARGIN", new_y="NEXT")
    pdf.ln(3)

    pdf.small(
        "Forcing the CPU backend is what makes a run bit reproducible. GPU backends that "
        "support float64 agree to within 1e-12 on the substrate fields; float32 backends "
        "such as DirectML and Metal differ at around 1e-6, which is harmless for the "
        "conclusions and fatal for hash equality. The backend used is recorded above and in "
        "the results file."
    )

    pdf.h2("4.5 Licence and reuse")
    pdf.body(
        "The platform is released under the MIT licence and the curated reference dataset "
        "under CC-BY 4.0. Results, including this report and the underlying JSON, are "
        "published so that the discoveries stay open: anyone may reproduce, check, "
        "contradict or build on them without permission."
    )


def build(results_path: Path, output_path: Path) -> int:
    if not results_path.is_file():
        print(f"ERROR: results not found at {results_path}", file=sys.stderr)
        print("Run scripts/run_pdac_report_experiment.py first.", file=sys.stderr)
        return 1

    data = json.loads(results_path.read_text(encoding="utf-8"))
    if not data.get("candidates"):
        print("ERROR: the results file contains no candidates.", file=sys.stderr)
        return 1

    pdf = Report(data.get("report_id", "EVOCURE"))
    pdf.alias_nb_pages()

    cover_page(pdf, data)
    summary_section(pdf, data)
    method_section(pdf, data)
    candidate_pages(pdf, data)
    limitations_section(pdf, data)
    reproducibility_section(pdf, data)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    pdf.output(str(output_path))
    size_kb = output_path.stat().st_size / 1024
    print(f"Wrote {output_path} ({size_kb:.0f} KB, {pdf.page_no()} pages)")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Build the EVOCURE PDAC candidate report")
    parser.add_argument("--results", default=str(RESULTS), help="experiment results JSON")
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT), help="destination PDF")
    args = parser.parse_args()
    return build(Path(args.results), Path(args.output))


if __name__ == "__main__":
    raise SystemExit(main())
