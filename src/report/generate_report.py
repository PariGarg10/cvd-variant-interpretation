"""
Step 8: Final Interpretation.

Combines:
  - AI classification (label + confidence, from the trained model)
  - SHAP explanation (top contributing features)
  - Structural comparison (RMSD)
  - Stability prediction (ddG, where available)

...into one structured, human-readable report per variant, matching the
report fields specified in the methodology (Section 5.8).

Usage (example — wire in real values from earlier stages):
    python -m src.report.generate_report
"""
import json
from dataclasses import dataclass, asdict


@dataclass
class VariantReport:
    gene_symbol: str
    variant: str  # e.g. "p.Arg403Gln"
    predicted_pathogenicity: str  # "Pathogenic" | "Benign" | "Uncertain (VUS)"
    confidence_score: float
    key_features: list  # from SHAP, e.g. [{"feature": ..., "shap_value": ...}, ...]
    structural_rmsd: float | None
    predicted_ddg: float | None
    stability_interpretation: str
    plain_language_summary: str


def build_plain_language_summary(report: "VariantReport") -> str:
    direction = "increases" if report.confidence_score >= 0.5 else "decreases"
    top_feature = report.key_features[0]["feature"] if report.key_features else "unavailable"
    return (
        f"The model classifies {report.gene_symbol} {report.variant} as "
        f"{report.predicted_pathogenicity} with {report.confidence_score:.0%} confidence. "
        f"The strongest contributing feature was '{top_feature}'. "
        f"Structural analysis indicates: {report.stability_interpretation}."
    )


def make_report(
    gene_symbol: str,
    variant: str,
    predicted_label: str,
    confidence_score: float,
    key_features: list,
    structural_rmsd: float | None,
    predicted_ddg: float | None,
    stability_interpretation: str,
) -> VariantReport:
    report = VariantReport(
        gene_symbol=gene_symbol,
        variant=variant,
        predicted_pathogenicity=predicted_label,
        confidence_score=confidence_score,
        key_features=key_features,
        structural_rmsd=structural_rmsd,
        predicted_ddg=predicted_ddg,
        stability_interpretation=stability_interpretation,
        plain_language_summary="",
    )
    report.plain_language_summary = build_plain_language_summary(report)
    return report


def save_report(report: VariantReport, out_path):
    with open(out_path, "w") as f:
        json.dump(asdict(report), f, indent=2)
    print(f"Saved variant interpretation report to {out_path}")


if __name__ == "__main__":
    # Example / smoke test with placeholder values — replace with real
    # outputs wired in from train_classifier.py, shap_explain.py, and
    # stability_analysis.py once the full pipeline is connected end to end.
    example = make_report(
        gene_symbol="MYH7",
        variant="p.Arg403Gln",
        predicted_label="Pathogenic",
        confidence_score=0.91,
        key_features=[
            {"feature": "delta_charge", "shap_value": 0.42},
            {"feature": "in_annotated_domain", "shap_value": 0.31},
        ],
        structural_rmsd=1.8,
        predicted_ddg=None,
        stability_interpretation="modest structural deviation (RMSD=1.80 \u00c5); ddG not available",
    )
    print(example.plain_language_summary)
    from src.config import OUTPUTS_DIR, ensure_dirs

    ensure_dirs()
    save_report(example, OUTPUTS_DIR / "example_variant_report.json")
