"""
Step 1 (part A): Pull ClinVar variant records for each gene in the panel.

Uses NCBI E-utilities (esearch -> esummary/efetch) against the ClinVar
database. This is free and does not require authentication, but you should
set NCBI_CONTACT_EMAIL in src/config.py, and ideally an API key, before
running this at scale (see NCBI's usage guidelines).

All matching records are retrieved: the esearch step pages through the
full result set rather than taking only the first page.

Usage:
    python -m src.data_collection.fetch_clinvar
"""
import time
import json
import requests
import pandas as pd
from tqdm import tqdm

from src.config import RAW_DIR, load_gene_panel, NCBI_CONTACT_EMAIL, NCBI_API_KEY, ensure_dirs

EUTILS_BASE = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
REQUEST_DELAY = 0.34  # ~3 requests/sec without an API key; use ~0.1 with a key


def _eutils_params(extra: dict) -> dict:
    params = {"retmode": "json", "email": NCBI_CONTACT_EMAIL, **extra}
    if NCBI_API_KEY:
        params["api_key"] = NCBI_API_KEY
    return params


def search_clinvar_ids(gene_symbol: str, page_size: int = 500) -> list[str]:
    """Return ALL ClinVar UIDs for missense, single-nucleotide variants in a gene.

    esearch returns at most `retmax` ids per call, so we page with `retstart`
    until the reported total is reached. (An earlier version requested a
    single page of 500, which silently truncated large genes such as MYH7.)
    """
    query = f'{gene_symbol}[gene] AND "single nucleotide variant"[Type] AND "missense variant"[molecular consequence]'
    ids: list[str] = []
    total = None
    while total is None or len(ids) < total:
        resp = requests.get(
            f"{EUTILS_BASE}/esearch.fcgi",
            params=_eutils_params(
                {"db": "clinvar", "term": query, "retmax": page_size, "retstart": len(ids)}
            ),
            headers={"User-Agent": "cvd-xai-project/0.1"},
            timeout=30,
        )
        resp.raise_for_status()
        result = resp.json().get("esearchresult", {})
        total = int(result.get("count", 0))
        page = result.get("idlist", [])
        if not page:
            break
        ids.extend(page)
        time.sleep(REQUEST_DELAY)
    return ids


def fetch_clinvar_summaries(uids: list[str]) -> list[dict]:
    """Fetch ClinVar summary records with robust clinical-significance parsing."""
    if not uids:
        return []

    resp = requests.get(
        f"{EUTILS_BASE}/esummary.fcgi",
        params=_eutils_params({
            "db": "clinvar",
            "id": ",".join(uids),
        }),
        headers={"User-Agent": "cvd-xai-project/0.1"},
        timeout=30,
    )
    resp.raise_for_status()

    result = resp.json().get("result", {})
    records = []

    for uid in result.get("uids", []):
        rec = result.get(uid, {})

        clinical = rec.get("clinical_significance")
        clinical_description = None
        review_status = None

        # Current/expected ClinVar format
        if isinstance(clinical, dict):
            clinical_description = clinical.get("description")
            review_status = clinical.get("review_status")

        # Handle possible list-based response formats
        elif isinstance(clinical, list):
            descriptions = []
            reviews = []

            for item in clinical:
                if isinstance(item, dict):
                    if item.get("description"):
                        descriptions.append(str(item["description"]))
                    if item.get("review_status"):
                        reviews.append(str(item["review_status"]))

            if descriptions:
                clinical_description = "; ".join(dict.fromkeys(descriptions))

            if reviews:
                review_status = "; ".join(dict.fromkeys(reviews))

        # Handle a plain string
        elif isinstance(clinical, str):
            clinical_description = clinical

        records.append(
            {
                "clinvar_uid": uid,
                "title": rec.get("title"),
                "gene_sort": rec.get("gene_sort"),
                "clinical_significance": (
    rec.get("germline_classification") or {}
).get("description"),
                "review_status": (
    rec.get("germline_classification") or {}
).get("review_status"),
                "variation_set": rec.get("variation_set"),
                "protein_change": rec.get("protein_change"),
                "molecular_consequence_list": rec.get(
                    "molecular_consequence_list"
                ),
                # Genes ClinVar itself links the record to; used downstream to
                # drop records pulled in only by an overlapping locus.
                "record_genes": ";".join(
                    g.get("symbol", "") for g in rec.get("genes", []) if isinstance(g, dict)
                ),
            }
        )

    return records


def fetch_gene_variants(gene_symbol: str) -> pd.DataFrame:
    uids = search_clinvar_ids(gene_symbol)
    time.sleep(REQUEST_DELAY)
    records = []
    # esummary in batches of 200 UIDs to stay well under URL length limits
    for i in range(0, len(uids), 200):
        batch = uids[i : i + 200]
        records.extend(fetch_clinvar_summaries(batch))
        time.sleep(REQUEST_DELAY)
    df = pd.DataFrame(records)
    if not df.empty:
        df["gene_symbol"] = gene_symbol
    return df


def main():
    ensure_dirs()
    genes = load_gene_panel("both")
    all_records = []
    for gene in tqdm(genes, desc="Fetching ClinVar records"):
        try:
            df = fetch_gene_variants(gene["symbol"])
            df["panel"] = gene["panel"]
            all_records.append(df)
        except requests.RequestException as e:
            print(f"[warn] failed to fetch {gene['symbol']}: {e}")

    if all_records:
        combined = pd.concat(all_records, ignore_index=True)
    else:
        combined = pd.DataFrame()

    out_path = RAW_DIR / "clinvar_raw.csv"
    combined.to_csv(out_path, index=False)
    print(f"Saved {len(combined)} ClinVar records to {out_path}")


if __name__ == "__main__":
    main()
