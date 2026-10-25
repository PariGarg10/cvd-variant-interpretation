"""
Streamlit front end for the CVD variant interpretation pipeline.

Every number shown here is read from the pipeline's outputs (data/ and
outputs/), so the app always reflects the latest run.

    streamlit run app.py
"""
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st

try:
    from streamlit_molstar import st_molstar
except ImportError:  # viewer is optional; the rest of the app works without it
    st_molstar = None

BASE_DIR = Path(__file__).resolve().parent
DATA = BASE_DIR / "data"
OUT = BASE_DIR / "outputs"

st.set_page_config(page_title="CVD Variant Interpretation", page_icon="🧬", layout="wide")


# ==================================================
# DATA
# ==================================================

@st.cache_data
def load_csv(path: Path, **kwargs) -> pd.DataFrame:
    return pd.read_csv(path, **kwargs) if path.exists() else pd.DataFrame()


@st.cache_data
def load_json(path: Path) -> dict:
    return json.load(open(path)) if path.exists() else {}


features = load_csv(DATA / "features" / "feature_matrix.csv")
predictions = load_csv(OUT / "logo_predictions.csv")
vus = load_csv(OUT / "vus_scores.csv")
metrics = load_json(OUT / "metrics.json")
raw_count = len(load_csv(DATA / "raw" / "clinvar_raw.csv", usecols=["clinvar_uid"]))

if not features.empty and not predictions.empty:
    features = features.merge(
        predictions[["clinvar_uid", "logo_RandomForest", "logo_XGBoost"]], on="clinvar_uid", how="left"
    )

FEATURE_LABELS = {
    "relative_position": "Relative position",
    "delta_hydrophobicity": "Δ hydrophobicity (Kyte–Doolittle)",
    "delta_molecular_weight": "Δ molecular weight (g/mol)",
    "delta_volume": "Δ side-chain volume (Å³)",
    "delta_charge": "Δ charge",
    "blosum62": "BLOSUM62 score",
    "conservation_score": "Conservation (fraction of orthologs)",
    "plddt": "AlphaFold pLDDT",
    "rsa": "Relative solvent accessibility",
    "contact_count": "Cα contacts within 10 Å",
    "polarity_change": "Polarity class change",
    "in_annotated_domain": "In annotated domain",
    "domain_type": "Domain feature type",
    "alphamissense": "AlphaMissense (comparator)",
}


# ==================================================
# LAYOUT
# ==================================================

st.sidebar.title("🧬 CVD Variant AI")
page = st.sidebar.radio(
    "Navigation", ["Dashboard", "Variant Explorer", "Model & SHAP", "VUS Prioritisation", "Structures"]
)
st.sidebar.caption("Research prototype — not for clinical use.")

st.title("Explainable AI for Cardiovascular Variant Interpretation")


# ==================================================
# DASHBOARD
# ==================================================
if page == "Dashboard":
    st.header("Project Dashboard")
    if features.empty:
        st.error("Feature matrix not found. Run `python -m src.pipeline --stage all` first.")
        st.stop()

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("ClinVar records", f"{raw_count:,}")
    c2.metric("Labelled variants", f"{len(features):,}")
    c3.metric("VUS", f"{len(vus):,}" if not vus.empty else "—")
    c4.metric("Genes", features["gene_symbol"].nunique())
    if metrics:
        best = metrics["best_model"]
        c5.metric(f"LOGO ROC-AUC ({best})", f"{metrics['logo'][best]['roc_auc']:.3f}")

    st.divider()
    left, right = st.columns([3, 2])
    with left:
        st.subheader("Labelled variants per gene")
        counts = pd.crosstab(features["gene_symbol"], features["label"])
        counts = counts.loc[counts.sum(1).sort_values(ascending=False).index]
        st.bar_chart(counts, color=["#3E6E8E", "#B5473A"])
    with right:
        st.subheader("Pipeline")
        st.markdown(
            """
            1. **Collect** — ClinVar, UniProt, AlphaFold DB, AlphaMissense
            2. **Preprocess** — parse, verify against UniProt, label, split off VUS
            3. **Features** — physicochemical, BLOSUM62, conservation, domain, structure
            4. **Classify** — Random Forest / XGBoost, leave-one-gene-out CV
            5. **Explain** — SHAP global and per-variant
            6. **Structure** — mutant side chain + OpenMM relaxation
            7. **Assess** — RMSD, burial, lost interactions
            8. **Interpret** — this app and the variant report
            """
        )


# ==================================================
# VARIANT EXPLORER
# ==================================================
elif page == "Variant Explorer":
    st.header("🔎 Variant Explorer")
    if features.empty:
        st.error("Feature matrix not found.")
        st.stop()

    gene = st.selectbox("Gene", sorted(features["gene_symbol"].unique()))
    gene_data = features[features["gene_symbol"] == gene].sort_values("position")
    label_filter = st.multiselect("ClinVar class", ["pathogenic", "benign"], default=["pathogenic", "benign"])
    gene_data = gene_data[gene_data["label"].isin(label_filter)]
    st.write(f"**{len(gene_data)} labelled variants in {gene}**")
    view = gene_data[["wt_aa", "position", "mut_aa", "label", "review_stars", "logo_RandomForest",
                      "alphamissense", "conservation_score", "plddt", "rsa"]].rename(
        columns={"logo_RandomForest": "model score (LOGO)"}
    )
    st.dataframe(view, use_container_width=True, hide_index=True)

    st.divider()
    idx = st.selectbox(
        "Inspect a variant",
        gene_data.index,
        format_func=lambda i: f"p.{gene_data.loc[i, 'wt_aa']}{gene_data.loc[i, 'position']}{gene_data.loc[i, 'mut_aa']}"
                              f"  ({gene_data.loc[i, 'label']})",
    )
    v = gene_data.loc[idx]
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Variant", f"{v['gene_symbol']} p.{v['wt_aa']}{v['position']}{v['mut_aa']}")
    c2.metric("ClinVar", f"{v['label']} ({int(v['review_stars'])}★)")
    c3.metric("Model score (unseen gene)", f"{v['logo_RandomForest']:.2f}" if pd.notna(v.get("logo_RandomForest")) else "—")
    c4.metric("AlphaMissense", f"{v['alphamissense']:.2f}" if pd.notna(v["alphamissense"]) else "—")

    left, right = st.columns(2)
    with left:
        st.subheader("Features")
        rows = [(FEATURE_LABELS[c], v[c]) for c in FEATURE_LABELS if c in v.index]
        st.dataframe(pd.DataFrame(rows, columns=["Feature", "Value"]).astype(str),
                     use_container_width=True, hide_index=True)
    with right:
        st.subheader("Why the model scores it this way (SHAP)")
        model_path = OUT / "model_random_forest.joblib"
        if model_path.exists():
            import shap
            from src.explainability.shap_explain import compute_shap_values
            from src.modeling.train_classifier import FEATURE_SETS, FULL_SET, columns

            model = joblib.load(model_path)
            explanation = compute_shap_values(model, gene_data.loc[[idx], columns(FEATURE_SETS[FULL_SET])])
            contrib = pd.DataFrame({"feature": explanation.feature_names, "shap": explanation.values[0]})
            contrib = contrib.reindex(contrib["shap"].abs().sort_values(ascending=False).index).head(8)
            st.bar_chart(contrib.set_index("feature")["shap"], horizontal=True)
            st.caption("Positive values push towards pathogenic, negative towards benign. "
                       "Values from the final model trained on all labelled variants.")


# ==================================================
# MODEL & SHAP
# ==================================================
elif page == "Model & SHAP":
    st.header("🤖 Model Performance & Explainability")
    if not metrics:
        st.error("metrics.json not found. Run the train stage.")
        st.stop()

    st.subheader("Leave-one-gene-out cross-validation")
    st.caption("Each gene is scored by a model that never saw any variant from that gene.")
    rows = []
    for name, m in [("Random Forest", metrics["logo"]["RandomForest"]), ("XGBoost", metrics["logo"]["XGBoost"]),
                    ("AlphaMissense (comparator)", metrics["comparators"]["AlphaMissense"])]:
        rows.append({"Model": name, "ROC-AUC": m["roc_auc"], "95% CI": str(m["roc_auc_ci"]), "PR-AUC": m["pr_auc"],
                     "Sensitivity": m["recall"], "Specificity": m["specificity"],
                     "Balanced acc.": m["balanced_accuracy"], "MCC": m["mcc"]})
    st.dataframe(pd.DataFrame(rows).round(3), use_container_width=True, hide_index=True)

    left, right = st.columns(2)
    with left:
        st.subheader("Feature-set ablation (ROC-AUC)")
        abl = pd.DataFrame({k: {m: v[m]["roc_auc"] for m in v} for k, v in metrics["ablation"].items()}).T
        st.dataframe(abl.round(3), use_container_width=True)
    with right:
        st.subheader("Leaky vs gene-aware evaluation")
        rc = metrics["random_cv"]["Full (+ structure)"]["RandomForest"]["roc_auc"]
        lg = metrics["logo"]["RandomForest"]["roc_auc"]
        st.metric("Random 5-fold CV ROC-AUC", f"{rc:.3f}", delta=f"{rc - lg:+.3f} vs LOGO", delta_color="off")
        st.metric("Leave-one-gene-out ROC-AUC", f"{lg:.3f}")

    st.divider()
    st.subheader("SHAP explainability")
    c1, c2 = st.columns(2)
    for col, name, cap in [(c1, "shap_global_summary.png", "Global SHAP summary"),
                           (c2, "shap_bar.png", "Mean |SHAP| per feature")]:
        if (OUT / name).exists():
            col.image(str(OUT / name), caption=cap, use_container_width=True)
    if (OUT / "shap_dependence.png").exists():
        st.image(str(OUT / "shap_dependence.png"), caption="Dependence plots for the top three features",
                 use_container_width=True)


# ==================================================
# VUS PRIORITISATION
# ==================================================
elif page == "VUS Prioritisation":
    st.header("📋 Variants of Uncertain Significance")
    if vus.empty:
        st.error("vus_scores.csv not found. Run the train stage.")
        st.stop()
    st.caption("VUS ranked by the final model. Scores are for prioritising manual review, not classification.")
    gene = st.selectbox("Gene", ["All"] + sorted(vus["gene_symbol"].unique()))
    sub = vus if gene == "All" else vus[vus["gene_symbol"] == gene]
    agree = st.checkbox("Only where the model and AlphaMissense agree (both ≥ 0.8)")
    if agree:
        sub = sub[(sub["model_score"] >= 0.8) & (sub["alphamissense"] >= 0.8)]
    c1, c2, c3 = st.columns(3)
    c1.metric("VUS shown", f"{len(sub):,}")
    c2.metric("Model score ≥ 0.8", f"{(sub['model_score'] >= 0.8).sum():,}")
    c3.metric("Model score < 0.2", f"{(sub['model_score'] < 0.2).sum():,}")
    cols = ["gene_symbol", "wt_aa", "position", "mut_aa", "review_stars", "model_score", "alphamissense",
            "conservation_score", "plddt", "rsa"]
    st.dataframe(sub.sort_values("model_score", ascending=False)[cols].head(500).round(3),
                 use_container_width=True, hide_index=True)
    st.bar_chart(np.histogram(sub["model_score"], bins=20, range=(0, 1))[0])


# ==================================================
# STRUCTURES
# ==================================================
elif page == "Structures":
    st.header("🧬 Protein Structure Analysis")
    analyses = sorted((OUT / "structures").glob("*_analysis.json"))
    if not analyses:
        st.error("No structural analyses found. Run `python -m src.structure.stability_analysis`.")
        st.stop()
    choice = st.selectbox("Case study", analyses, format_func=lambda p: p.stem.replace("_analysis", "").replace("_", " p."))
    a = json.load(open(choice))
    tag = f"{a['gene']}_{a['variant']}"

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Variant", f"{a['gene']} p.{a['variant']}")
    c2.metric("Site RSA (WT → mutant)", f"{a['wt']['rsa']:.2f} → {a['mutant']['rsa']:.2f}")
    c3.metric("Local heavy-atom RMSD", f"{a['rmsd']['local_heavy_rmsd']:.2f} Å")
    c4.metric("ΔΔG", "n/a" if a["ddg"] is None else f"{a['ddg']:.2f}")
    st.info(a["interpretation"])

    left, right = st.columns(2)
    for col, name in ((left, "wt"), (right, "mutant")):
        with col:
            st.subheader("Wild type" if name == "wt" else "Mutant")
            st.write("**H-bonds:** " + (", ".join(a[name]["hbonds"]) or "none"))
            st.write("**Salt bridges:** " + (", ".join(a[name]["salt_bridges"]) or "none"))
            path = OUT / "structures" / f"{tag}_{name}_full.pdb"
            if st_molstar is not None and path.exists():
                st_molstar(str(path), key=f"{tag}_{name}", height=420)
    st.caption("Viewer: full-length AlphaFold DB model (UniProt numbering) with the mutant side chain built in. "
               "Metrics above come from the region within 20 Å of the site after OpenMM relaxation (Amber14/OBC2).")
