import streamlit as st
import pandas as pd
from pathlib import Path
from streamlit_molstar import st_molstar


# ==================================================
# PAGE CONFIG
# ==================================================

st.set_page_config(
    page_title="CVD Variant Interpretation",
    page_icon="🧬",
    layout="wide"
)


# ==================================================
# PATHS
# ==================================================

BASE_DIR = Path(__file__).resolve().parent

FEATURE_FILE = (
    BASE_DIR
    / "data"
    / "features"
    / "feature_matrix.csv"
)

STRUCTURE_FILE = (
    BASE_DIR
    / "data"
    / "raw"
    / "structure_manifest.csv"
)


# ==================================================
# LOAD DATA
# ==================================================

@st.cache_data
def load_features():

    if FEATURE_FILE.exists():
        return pd.read_csv(FEATURE_FILE)

    return pd.DataFrame()


@st.cache_data
def load_structures():

    if STRUCTURE_FILE.exists():
        return pd.read_csv(STRUCTURE_FILE)

    return pd.DataFrame()


features = load_features()
structures = load_structures()


# ==================================================
# SIDEBAR
# ==================================================

st.sidebar.title("🧬 CVD Variant AI")

page = st.sidebar.radio(
    "Navigation",
    [
        "Dashboard",
        "Variant Explorer",
        "ML & SHAP",
        "Structures"
    ]
)


# ==================================================
# HEADER
# ==================================================

st.title(
    "Explainable AI for Cardiovascular Variant Interpretation"
)

st.markdown(
    """
    An AI-assisted platform for exploring cardiovascular genetic variants
    using clinical annotations, biological features, machine learning,
    conservation analysis and protein structures.
    """
)


# ==================================================
# DASHBOARD
# ==================================================

if page == "Dashboard":

    st.header("Project Dashboard")

    total_variants = 7688

    labelled_variants = len(features)

    if not features.empty and "label" in features.columns:

        pathogenic = int(
            (features["label"] == 1).sum()
        )

        benign = int(
            (features["label"] == 0).sum()
        )

    else:

        pathogenic = 0
        benign = 0

    if not structures.empty:

        structure_count = len(structures)

    else:

        structure_count = 18


    # --------------------------------------------------
    # METRICS
    # --------------------------------------------------

    col1, col2, col3, col4 = st.columns(4)

    with col1:

        st.metric(
            "Variants Collected",
            f"{total_variants:,}"
        )

    with col2:

        st.metric(
            "Labelled Variants",
            f"{labelled_variants:,}"
        )

    with col3:

        st.metric(
            "Genes",
            "18"
        )

    with col4:

        st.metric(
            "Structures",
            structure_count
        )


    st.divider()


    # --------------------------------------------------
    # CLINICAL LABELS
    # --------------------------------------------------

    col1, col2 = st.columns(2)

    with col1:

        st.subheader("Clinical Labels")

        label_data = pd.DataFrame(
            {
                "Classification": [
                    "Pathogenic",
                    "Benign"
                ],

                "Variants": [
                    pathogenic,
                    benign
                ]
            }
        )

        st.bar_chart(
            label_data.set_index(
                "Classification"
            )
        )


    # --------------------------------------------------
    # PIPELINE
    # --------------------------------------------------

    with col2:

        st.subheader("Pipeline")

        st.markdown(
            """
            **1. Data Collection**

            ClinVar + UniProt + PDB

            ↓

            **2. Preprocessing**

            ↓

            **3. Feature Engineering**

            ↓

            **4. Conservation Analysis**

            ↓

            **5. Machine Learning**

            ↓

            **6. SHAP Explainability**

            ↓

            **7. Protein Structure Analysis**

            ↓

            **8. Variant Interpretation**
            """
        )


# ==================================================
# VARIANT EXPLORER
# ==================================================

elif page == "Variant Explorer":

    st.header("🔎 Variant Explorer")


    if features.empty:

        st.error(
            "Feature matrix not found. "
            "Run the preprocessing and feature stages first."
        )


    else:

        # --------------------------------------------------
        # GENE SELECTION
        # --------------------------------------------------

        genes = sorted(
            features[
                "gene_symbol"
            ]
            .dropna()
            .unique()
            .tolist()
        )


        selected_gene = st.selectbox(
            "Select Gene",
            genes
        )


        gene_data = features[
            features["gene_symbol"]
            == selected_gene
        ]


        st.write(
            f"**{len(gene_data)} variants found "
            f"for {selected_gene}**"
        )


        # --------------------------------------------------
        # VARIANT TABLE
        # --------------------------------------------------

        st.dataframe(
            gene_data,
            use_container_width=True
        )


        st.divider()


        # --------------------------------------------------
        # VARIANT DETAILS
        # --------------------------------------------------

        st.subheader("Variant Details")


        variant_index = st.selectbox(
            "Select a variant",
            gene_data.index,

            format_func=lambda x:
                f"{gene_data.loc[x, 'wt_aa']}"
                f"{gene_data.loc[x, 'position']}"
                f"{gene_data.loc[x, 'mut_aa']}"
        )


        variant = gene_data.loc[
            variant_index
        ]


        # --------------------------------------------------
        # BASIC VARIANT INFORMATION
        # --------------------------------------------------

        col1, col2, col3, col4 = st.columns(4)


        with col1:

            st.metric(
                "Gene",
                variant["gene_symbol"]
            )


        with col2:

            st.metric(
                "Position",
                int(variant["position"])
            )


        with col3:

            st.metric(
                "Wild Type",
                variant["wt_aa"]
            )


        with col4:

            st.metric(
                "Mutation",
                variant["mut_aa"]
            )


        # --------------------------------------------------
        # BIOLOGICAL FEATURES
        # --------------------------------------------------

        st.subheader(
            "Biological Features"
        )


        feature_cols = [

            "delta_hydrophobicity",

            "delta_molecular_weight",

            "delta_volume",

            "delta_charge",

            "polarity_change",

            "conservation_score",

            "in_annotated_domain",

            "domain_type"
        ]


        available = [

            c

            for c in feature_cols

            if c in variant.index
        ]


        feature_table = pd.DataFrame(
            {
                "Feature": available,

                "Value": [
                    variant[c]

                    for c in available
                ]
            }
        )


        st.dataframe(
            feature_table,

            use_container_width=True,

            hide_index=True
        )


# ==================================================
# ML & SHAP
# ==================================================

elif page == "ML & SHAP":

    st.header(
        "🤖 Machine Learning & Explainability"
    )


    # --------------------------------------------------
    # MODEL PERFORMANCE
    # --------------------------------------------------

    st.subheader(
        "Model Performance"
    )


    col1, col2 = st.columns(2)


    # --------------------------------------------------
    # RANDOM FOREST
    # --------------------------------------------------

    with col1:

        st.markdown(
            "### Random Forest"
        )


        rf_metrics = pd.DataFrame(
            {
                "Metric": [

                    "Accuracy",

                    "Precision",

                    "Recall",

                    "F1",

                    "ROC-AUC",

                    "PR-AUC"
                ],


                "Score": [

                    0.622,

                    0.647,

                    0.887,

                    0.748,

                    0.588,

                    0.706
                ]
            }
        )


        st.dataframe(
            rf_metrics,

            use_container_width=True,

            hide_index=True
        )


    # --------------------------------------------------
    # XGBOOST
    # --------------------------------------------------

    with col2:

        st.markdown(
            "### XGBoost"
        )


        xgb_metrics = pd.DataFrame(
            {
                "Metric": [

                    "Accuracy",

                    "Precision",

                    "Recall",

                    "F1",

                    "ROC-AUC",

                    "PR-AUC"
                ],


                "Score": [

                    0.633,

                    0.686,

                    0.774,

                    0.727,

                    0.559,

                    0.703
                ]
            }
        )


        st.dataframe(
            xgb_metrics,

            use_container_width=True,

            hide_index=True
        )


    st.divider()


    # --------------------------------------------------
    # SHAP
    # --------------------------------------------------

    st.subheader(
        "SHAP Explainability"
    )


    st.info(
        """
        SHAP analysis identifies which biological features
        contribute most strongly to model predictions.
        """
    )


    shap_candidates = [

        BASE_DIR
        / "outputs"
        / "shap_summary.png",

        BASE_DIR
        / "outputs"
        / "shap_summary_plot.png",

        BASE_DIR
        / "outputs"
        / "shap_global_summary.png"
    ]


    shap_file = next(
        (
            p

            for p in shap_candidates

            if p.exists()
        ),

        None
    )


    if shap_file:

        st.image(
            str(shap_file),

            caption="Global SHAP Feature Importance",

            use_container_width=True
        )


    else:

        st.warning(
            """
            SHAP plot not found in outputs/.
            Add the generated SHAP plot there.
            """
        )


# ==================================================
# STRUCTURES
# ==================================================

elif page == "Structures":

    st.header(
        "🧬 Protein Structure Analysis"
    )


    st.markdown(
        """
        Interactive visualization of the wild-type and mutant
        protein structures for the selected variant.
        """
    )


    # --------------------------------------------------
    # CURRENT VARIANT
    # --------------------------------------------------

    st.subheader(
        "Current Variant"
    )


    col1, col2, col3, col4 = st.columns(4)


    with col1:

        st.metric(
            "Gene",
            "ACTC1"
        )


    with col2:

        st.metric(
            "Variant",
            "R39H"
        )


    with col3:

        st.metric(
            "PDB",
            "9B3R"
        )


    with col4:

        st.metric(
            "Chain",
            "A"
        )


    st.divider()


    # --------------------------------------------------
    # FILE PATHS
    # --------------------------------------------------

    wt_file = (

        BASE_DIR

        / "outputs"

        / "ACTC1_R39H_wt.pdb"
    )


    mutant_file = (

        BASE_DIR

        / "outputs"

        / "ACTC1_R39H_mutant.pdb"
    )


    # --------------------------------------------------
    # TWO STRUCTURES
    # --------------------------------------------------

    col1, col2 = st.columns(2)


    # ==================================================
    # WT
    # ==================================================

    with col1:

        st.subheader(
            "🧬 Wild-Type"
        )


        st.caption(
            "Experimental structure — PDB 9B3R"
        )


        if wt_file.exists():

            st_molstar(
                str(wt_file)
            )


            st.success(
                "WT structure loaded successfully"
            )


            st.info(
                "Chain A • Residue 39 = "
                "Arginine (R)"
            )


        else:

            st.error(
                f"WT structure not found:\n"
                f"{wt_file}"
            )


    # ==================================================
    # MUTANT
    # ==================================================

    with col2:

        st.subheader(
            "🧬 R39H Mutant"
        )


        st.caption(
            "Computational mutant model"
        )


        if mutant_file.exists():

            st_molstar(
                str(mutant_file)
            )


            st.success(
                "R39H mutant structure "
                "loaded successfully"
            )


            st.info(
                "Chain A • Residue 39 = "
                "Histidine (H)"
            )


        else:

            st.error(
                f"Mutant structure not found:\n"
                f"{mutant_file}"
            )


    st.divider()


    # --------------------------------------------------
    # STRUCTURAL ANALYSIS
    # --------------------------------------------------

    st.subheader(
        "📊 Structural Analysis"
    )


    col1, col2 = st.columns(2)


    with col1:

        st.metric(
            "Cα RMSD",
            "0.00 Å"
        )


    with col2:

        st.metric(
            "ΔΔG",
            "Not available"
        )


    st.warning(
        """
        The current mutant model changes the residue identity
        but does not perform side-chain rebuilding or structural
        relaxation.

        Therefore, the 0.00 Å RMSD should not be interpreted as
        evidence that R39H has no structural effect.
        """
    )


    # --------------------------------------------------
    # STRUCTURE FILE STATUS
    # --------------------------------------------------

    st.subheader(
        "📁 Structure Files"
    )


    structure_status = pd.DataFrame(
        {

            "Structure": [

                "Wild Type",

                "R39H Mutant"
            ],


            "PDB": [

                "9B3R",

                "9B3R → R39H"
            ],


            "File": [

                wt_file.name,

                mutant_file.name
            ],


            "Status": [

                "Available"
                if wt_file.exists()
                else "Missing",

                "Available"
                if mutant_file.exists()
                else "Missing"
            ]
        }
    )


    st.dataframe(

        structure_status,

        use_container_width=True,

        hide_index=True
    )