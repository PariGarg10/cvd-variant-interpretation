"""
Top-level orchestrator. Runs pipeline stages in order via CLI flags, so you
can run the whole thing or resume from any stage during development.

Usage:
    python -m src.pipeline --stage all
    python -m src.pipeline --stage collect
    python -m src.pipeline --stage preprocess
    python -m src.pipeline --stage features
    python -m src.pipeline --stage train
    python -m src.pipeline --stage explain
    python -m src.pipeline --stage structure
"""
import argparse

from src.config import ensure_dirs


def run_collect():
    from src.data_collection import (
        fetch_alphafold,
        fetch_alphamissense,
        fetch_clinvar,
        fetch_structures,
        fetch_uniprot,
    )

    fetch_clinvar.main()
    fetch_uniprot.main()
    fetch_structures.main()
    fetch_alphafold.main()
    fetch_alphamissense.main()


def run_preprocess():
    from src.preprocessing import clean_variants

    clean_variants.main()


def run_features():
    from src.features import build_conservation_scores, build_feature_matrix, structural_features

    build_conservation_scores.main()
    structural_features.main()
    build_feature_matrix.main()


def run_train():
    from src.modeling import train_classifier

    train_classifier.main()


def run_explain():
    from src.explainability import shap_explain

    shap_explain.main()


def run_structure():
    from src.structure import stability_analysis

    stability_analysis.main()


STAGES = {
    "collect": run_collect,
    "preprocess": run_preprocess,
    "features": run_features,
    "train": run_train,
    "explain": run_explain,
    "structure": run_structure,
}


def main():
    parser = argparse.ArgumentParser(description="Cardiovascular Variant XAI pipeline")
    parser.add_argument(
        "--stage",
        choices=list(STAGES.keys()) + ["all"],
        default="all",
        help="Which stage to run (default: all, in order)",
    )
    args = parser.parse_args()

    ensure_dirs()

    if args.stage == "all":
        for name, fn in STAGES.items():
            print(f"\n{'=' * 60}\nSTAGE: {name}\n{'=' * 60}")
            fn()
    else:
        STAGES[args.stage]()


if __name__ == "__main__":
    main()
