Absolutely. Since you have now **actually run the live data collection, implemented conservation, trained the models, generated SHAP, and completed the ACTC1 R39H structure proof-of-concept + Streamlit frontend**, the README should no longer say those things are “not yet run.”

Here is the **updated README** you can replace the current one with:

````markdown
# Explainable AI for Cardiovascular Variant Interpretation

An explainable AI pipeline for prioritizing and interpreting cardiovascular genetic variants using clinical annotations, biological features, machine learning, conservation analysis, SHAP explainability, protein structures, and stability analysis.

The project follows an 8-step workflow:

data collection → preprocessing → feature engineering → ML classification → SHAP explainability → structure analysis → stability assessment → integrated interpretation

---

## Status

This is a **working MVP / research prototype** implementing the complete pipeline architecture.

Steps **1–5 have been implemented and executed on the collected dataset**:

- ClinVar variant collection
- UniProt protein sequence retrieval
- Protein structure metadata collection
- ClinVar preprocessing and protein-change parsing
- Pathogenic/benign labelling
- VUS separation
- Physicochemical feature engineering
- Homolog-based conservation scoring
- Domain feature integration
- Gene-level train/test splitting
- Random Forest and XGBoost classification
- SHAP global explainability

Steps **6–8 have been implemented as a proof-of-concept on a curated variant**:

- Experimental PDB structure selection
- WT structure visualization
- Mutant structure generation
- Structural comparison
- RMSD calculation
- Stability-analysis framework
- Streamlit-based interactive frontend

The current structural proof-of-concept uses:

**ACTC1 R39H**

with experimental PDB structure:

**PDB: 9B3R**

The project also includes an interactive Streamlit interface with:

- Dashboard
- Variant Explorer
- ML & SHAP visualization
- Interactive 3D protein structure visualization using MolStar

---

# Pipeline

## Step 1 — Data Collection

Data is collected from public biological databases:

- **ClinVar** — clinical variant classifications
- **UniProt** — protein sequences and annotations
- **PDB** — experimental protein structures

The collected data is stored locally under:

```text
data/raw/
````

### Current collected data

* **7,688 ClinVar variants**
* **18 cardiovascular genes**
* **570 labelled variants**
* **6,122 VUS variants held out for qualitative review**
* **18 structure records**

ClinVar classifications include pathogenic, likely pathogenic, benign, likely benign, conflicting classifications, and variants of uncertain significance.

Before running data collection, configure:

```text
src/config.py
```

and set:

```python
NCBI_CONTACT_EMAIL = "your_email@example.com"
```

An NCBI API key can optionally be configured to increase the E-utilities rate limit.

---

# Step 2 — Preprocessing

The preprocessing stage:

1. Cleans the raw ClinVar data
2. Parses protein-level changes
3. Removes unusable records
4. Separates labelled variants from VUS
5. Produces a binary pathogenic/benign dataset

The parser supports both:

```text
M515V
```

and three-letter protein notation such as:

```text
Met515Val
```

### Current labelled dataset

```text
Total labelled variants: 570

Pathogenic: 424
Benign:     146
```

VUS variants are not discarded. They are retained separately in:

```text
data/processed/variants_vus.csv
```

for future qualitative analysis.

---

# Step 3 — Feature Engineering

The feature matrix combines physicochemical and biological characteristics of each variant.

Current features include:

```text
gene_symbol
panel
position
wt_aa
mut_aa
label

delta_hydrophobicity
delta_molecular_weight
delta_volume
delta_charge
polarity_change

wt_is_proline
mut_is_proline
wt_is_glycine
mut_is_glycine

conservation_score
in_annotated_domain
domain_type
```

The final feature matrix is stored at:

```text
data/features/feature_matrix.csv
```

Current matrix:

```text
570 variants × 18 features
```

---

# Conservation Analysis

Conservation analysis has been implemented using a transparent homolog-based approach.

For each cardiovascular gene:

1. Reviewed UniProt homologs are retrieved.
2. The human/reference sequence is excluded from the homolog set.
3. Homologous sequences are aligned using Biopython's `PairwiseAligner`.
4. The amino acid corresponding to each reference position is examined.
5. A conservation score is calculated as the fraction of selected homologs retaining the reference amino acid.

The resulting scores are stored in:

```text
data/raw/conservation_scores.json
```

and incorporated into:

```text
data/features/feature_matrix.csv
```

### Current conservation coverage

```text
Scored variants: 518
Missing scores:   52
```

Missing conservation values are handled using median imputation inside the machine-learning preprocessing pipeline.

### Important limitation

This is a **homolog-based conservation proxy**, not a formal phylogenetic conservation score such as ConSurf, phyloP, or a full evolutionary model.

The current implementation uses a relatively small number of reviewed homologs per gene and should therefore be interpreted as an explainable conservation feature rather than a clinically validated conservation predictor.

---

# Step 4 — Machine Learning

Two classification models are implemented:

* Random Forest
* XGBoost

A **gene-level train/test split** is used rather than a random row split.

This prevents variants from the same gene appearing in both training and testing datasets, reducing the risk of overly optimistic performance caused by gene-specific information leakage.

### Dataset split

```text
Training genes: 13
Training variants: 472

Testing genes: 5
Testing variants: 98
```

Class weighting is used instead of SMOTE by default.

This avoids generating synthetic biological variants during training.

---

# Model Performance

Current evaluation results after incorporating the conservation feature:

## Random Forest

| Metric    | Score |
| --------- | ----: |
| Accuracy  | 0.622 |
| Precision | 0.647 |
| Recall    | 0.887 |
| F1        | 0.748 |
| ROC-AUC   | 0.588 |
| PR-AUC    | 0.706 |

Confusion matrix:

```text
[[ 6, 30],
 [ 7, 55]]
```

---

## XGBoost

| Metric    | Score |
| --------- | ----: |
| Accuracy  | 0.633 |
| Precision | 0.686 |
| Recall    | 0.774 |
| F1        | 0.727 |
| ROC-AUC   | 0.559 |
| PR-AUC    | 0.703 |

Confusion matrix:

```text
[[14, 22],
 [14, 48]]
```

These results are intended as **research-prototype evaluation results**, not clinical performance claims.

The gene-level split makes the evaluation more conservative, but the dataset remains relatively small and the models should not be considered clinically validated.

---

# Step 5 — SHAP Explainability

SHAP is used to provide model-level feature explanations.

The SHAP analysis provides:

* Global feature importance
* Feature contribution visualization
* Per-variant explanations

The generated SHAP outputs are stored under:

```text
outputs/
```

The global analysis shows that features such as:

```text
position
delta_volume
delta_hydrophobicity
delta_molecular_weight
delta_charge
conservation_score
```

contribute substantially to the model's predictions.

The conservation feature is therefore incorporated not only into the predictive model but also into the explainability layer.

---

# Step 6 — Protein Structure Analysis

Protein structure analysis is implemented as a proof-of-concept for selected high-priority variants.

The original implementation included an ESMFold-based mutant structure prediction workflow.

The public ESMFold endpoint was tested but repeatedly returned gateway timeout errors during execution.

Therefore, the current proof-of-concept uses an **experimental PDB structure** for ACTC1.

### Example variant

```text
Gene:       ACTC1
Variant:    R39H
PDB:        9B3R
Chain:      A
Position:   39
WT residue: R (Arginine)
Mutant:     H (Histidine)
```

PDB 9B3R contains the relevant ACTC1 residue and was selected for the structural demonstration.

The wild-type structure is stored as:

```text
outputs/ACTC1_R39H_wt.pdb
```

The generated mutant structure is stored as:

```text
outputs/ACTC1_R39H_mutant.pdb
```

---

# Mutant Structure Generation

The mutant-generation script performs the amino-acid substitution:

```text
R39 → H
```

on chain A.

The current implementation changes the residue identity while retaining the existing backbone/atomic coordinates.

Therefore, this is a **structural mutation proof-of-concept**, rather than a fully relaxed mutant structure.

A production-quality structural workflow would additionally require:

* side-chain rebuilding
* rotamer optimization
* energy minimization
* structural relaxation

---

# Step 7 — Stability Assessment

The stability-analysis module currently supports:

* Cα RMSD calculation
* FoldX integration framework
* ΔΔG interpretation

The current ACTC1 R39H proof-of-concept produced:

```text
Cα RMSD = 0.00 Å
ΔΔG     = Not available
```

### Important interpretation

The 0.00 Å RMSD is expected because the current mutation procedure changes the residue identity without changing the atomic coordinates.

Therefore:

> **0.00 Å RMSD does NOT demonstrate that R39H has no structural or stability effect.**

It only indicates that the coordinates used for the WT and mutant structures are identical in the current proof-of-concept representation.

FoldX was not installed, so a quantitative ΔΔG value is currently unavailable.

This limitation is explicitly retained rather than assigning an unsupported stability prediction.

---

# Step 8 — Integrated Interpretation

The project is designed to combine:

```text
Clinical evidence
       +
Physicochemical properties
       +
Conservation
       +
Machine-learning prediction
       +
SHAP explanation
       +
Protein structure
       +
Structural/stability evidence
```

into an integrated variant interpretation.

The current implementation demonstrates this workflow on a curated structural variant subset rather than automatically generating structural interpretations for all thousands of variants.

---

# Streamlit Frontend

The project includes an interactive Streamlit application.

Run:

```bash
streamlit run app.py
```

The application contains four sections.

## Dashboard

Displays:

* Total collected variants
* Labelled variants
* Number of genes
* Available structures
* Clinical label distribution
* Pipeline overview

---

## Variant Explorer

Allows the user to:

* Select a gene
* Browse variants
* Inspect amino-acid substitutions
* View physicochemical features
* View conservation scores
* View domain annotations

---

## ML & SHAP

Displays:

* Random Forest performance
* XGBoost performance
* SHAP global feature importance
* Explainability outputs

---

## Structures

Provides an interactive 3D molecular viewer using **MolStar**.

The current demonstration displays:

```text
ACTC1 WT
```

and

```text
ACTC1 R39H
```

side-by-side using the PDB files generated by the structural pipeline.

---

# Setup

Create a virtual environment:

```bash
python -m venv venv
```

Activate it on Windows:

```bash
venv\Scripts\activate
```

On Linux/macOS:

```bash
source venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

For the Streamlit frontend, ensure:

```bash
pip install streamlit streamlit-molstar
```

---

# Running the Pipeline

Run each stage sequentially:

```bash
python -m src.pipeline --stage collect

python -m src.pipeline --stage preprocess

python -m src.pipeline --stage features

python -m src.pipeline --stage train

python -m src.pipeline --stage explain
```

Or run the first five stages together:

```bash
python -m src.pipeline --stage all
```

---

# Running Structural Analysis

Structural analysis is currently performed on selected variants.

For a variant with an available experimental structure:

```bash
python -m src.structure.mutate_pdb
```

The current demonstration generates:

```text
outputs/ACTC1_R39H_mutant.pdb
```

Then run:

```bash
python -m src.structure.stability_analysis
```

to calculate the available structural metrics.

---

# Tests

Run the unit tests using:

```bash
pytest tests/ -v
```

The current tests cover the physicochemical feature-engineering components, which do not require network access.

---

# Project Structure

```text
cvdproject/
│
├── app.py
│
├── config/
│   └── genes.yaml
│
├── data/
│   ├── raw/
│   │   ├── clinvar_raw.csv
│   │   ├── uniprot_entries.json
│   │   ├── structure_manifest.csv
│   │   └── conservation_scores.json
│   │
│   ├── processed/
│   │   ├── variants_labelled.csv
│   │   └── variants_vus.csv
│   │
│   └── features/
│       └── feature_matrix.csv
│
├── src/
│   │
│   ├── data_collection/
│   │   ├── fetch_clinvar.py
│   │   ├── fetch_uniprot.py
│   │   ├── fetch_structures.py
│   │   └── ...
│   │
│   ├── preprocessing/
│   │   └── ...
│   │
│   ├── features/
│   │   ├── physicochemical.py
│   │   ├── conservation_and_domain.py
│   │   ├── build_conservation_scores.py
│   │   └── build_feature_matrix.py
│   │
│   ├── modeling/
│   │   └── train_classifier.py
│   │
│   ├── explainability/
│   │   └── ...
│   │
│   ├── structure/
│   │   ├── predict_mutant_structure.py
│   │   ├── mutate_pdb.py
│   │   └── stability_analysis.py
│   │
│   ├── report/
│   │   └── ...
│   │
│   └── pipeline.py
│
├── outputs/
│   ├── trained models
│   ├── SHAP plots
│   ├── ACTC1_R39H_wt.pdb
│   ├── ACTC1_R39H_mutant.pdb
│   └── reports
│
├── tests/
│
├── requirements.txt
│
└── README.md
```

---

# Methodology Decisions

## Gene-level train/test split

Variants are split by gene rather than randomly by row.

This prevents the same gene from appearing in both training and testing datasets.

---

## Class weighting

Class weighting is used to address class imbalance rather than generating synthetic variants through SMOTE.

This keeps the training data composed of observed variants.

---

## VUS handling

Variants of uncertain significance are retained separately rather than being forced into the binary classification task.

This allows future qualitative analysis and avoids introducing artificial labels.

---

## Conservation methodology

The current conservation feature is a homolog-based identity score.

It is deliberately kept transparent and reproducible, while acknowledging that it is not equivalent to a clinically validated evolutionary conservation predictor.

---

## Structural methodology

Experimental PDB structures are preferred when an appropriate structure covering the mutation position is available.

The current structural mutation workflow is a proof-of-concept and does not perform full side-chain relaxation.

---

# Known Limitations and Future Work

The main remaining limitations are:

### 1. Improved conservation scoring

The current homolog-based conservation proxy could be replaced or supplemented with:

* ConSurf
* phyloP
* PhastCons
* larger ortholog MSAs
* evolutionary conservation models

---

### 2. Quantitative stability prediction

FoldX or another validated stability-prediction tool can be integrated to calculate mutation-induced ΔΔG.

The current implementation intentionally reports:

```text
ΔΔG = Not available
```

when no stability calculation has been performed.

---

### 3. Improved PDB structure ranking

The current structure-selection workflow can be improved by ranking structures according to:

* experimental resolution
* sequence coverage
* mutation-position coverage
* chain identity
* structural completeness

This is particularly important for proteins such as MYH7 that have multiple available structures.

---

### 4. Improved mutant modelling

The current mutation script changes residue identity without rebuilding the mutant side chain.

Future implementations could use:

* FoldX RepairPDB/BuildModel
* PyRosetta
* Modeller
* side-chain rotamer optimization
* energy minimization

---

### 5. Larger evaluation dataset

The current labelled dataset contains 570 variants.

Future work should evaluate the approach on larger independent datasets and compare against established variant-effect predictors.

---

### 6. Clinical validation

The system is a research prototype and is **not intended for clinical diagnosis or direct medical decision-making**.

Further validation, benchmarking, calibration, and independent testing would be required before any clinical application.

---

# Reproducibility

The project is designed so that the major computational stages can be reproduced locally.

After installing the dependencies and configuring the NCBI contact email:

```bash
python -m src.pipeline --stage collect
python -m src.pipeline --stage preprocess
python -m src.pipeline --stage features
python -m src.pipeline --stage train
python -m src.pipeline --stage explain
```

The Streamlit application can then be launched with:

```bash
streamlit run app.py
```

---

# Summary

This project demonstrates an end-to-end explainable AI workflow for cardiovascular variant interpretation:

```text
                 ┌──────────────────┐
                 │     ClinVar       │
                 └────────┬─────────┘
                          ↓
                 ┌──────────────────┐
                 │  Preprocessing   │
                 └────────┬─────────┘
                          ↓
                 ┌──────────────────┐
                 │ Feature Engineer │
                 └────────┬─────────┘
                          ↓
              ┌─────────────────────────┐
              │ Conservation + Biology │
              └────────────┬────────────┘
                           ↓
                 ┌──────────────────┐
                 │ Random Forest /  │
                 │    XGBoost       │
                 └────────┬─────────┘
                          ↓
                 ┌──────────────────┐
                 │      SHAP        │
                 └────────┬─────────┘
                          ↓
                 ┌──────────────────┐
                 │ Protein Structure│
                 └────────┬─────────┘
                          ↓
                 ┌──────────────────┐
                 │ Stability / RMSD │
                 └────────┬─────────┘
                          ↓
                 ┌──────────────────┐
                 │   Interpretation │
                 └──────────────────┘
                          ↓
                 ┌──────────────────┐
                 │ Streamlit +      │
                 │ MolStar Frontend │
                 └──────────────────┘
```

The current implementation provides a reproducible MVP combining **clinical variant data, biological feature engineering, machine learning, explainability, evolutionary information, structural analysis, and an interactive visualization interface**.

```

### One thing I would change from your old README

Don't write:

> “Everything that needs a live API is fully coded but not yet run”

anymore. You've actually run the live collection and generated real outputs.

Also, don't claim that the **ACTC1 R39H RMSD = 0 means no structural change**. The README above explicitly explains why that's not a valid biological conclusion—which is important if you're showing this project to a professor, recruiter, or during a viva.
```
