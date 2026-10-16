import pandas as pd

def load_data(anonymized_path, auxiliary_path):
    """
    Load anonymized and auxiliary datasets.
    """
    anon = pd.read_csv(anonymized_path)
    aux = pd.read_csv(auxiliary_path)
    return anon, aux


def link_records(anon_df, aux_df):
    """
    Attempt to link anonymized records to auxiliary records
    using exact matching on quasi-identifiers.

    Returns a DataFrame with columns:
      anon_id, matched_name
    containing ONLY uniquely matched records.
    """
    quasi_identifiers = ["age", "zip3", "gender"]
    missing_cols = [col for col in quasi_identifiers if col not in anon_df.columns or col not in aux_df.columns]
    if missing_cols:
        raise ValueError(f"Missing required quasi-identifier columns: {missing_cols}")

    if "name" not in aux_df.columns:
        raise ValueError("Auxiliary data must include a 'name' column.")

    aux_counts = (
        aux_df.groupby(quasi_identifiers, dropna=False)
        .size()
        .rename("match_count")
        .reset_index()
    )

    matches = anon_df.merge(aux_counts, on=quasi_identifiers, how="left")
    unique_matches = matches[matches["match_count"] == 1].copy()

    if unique_matches.empty:
        return pd.DataFrame(columns=["anon_id", "matched_name"])

    unique_matches = unique_matches.merge(
        aux_df[["name"] + quasi_identifiers],
        on=quasi_identifiers,
        how="left",
    )

    result = unique_matches[["anon_id", "name"]].rename(columns={"name": "matched_name"})
    return result.drop_duplicates().reset_index(drop=True)


def deanonymization_rate(matches_df, anon_df):
    """
    Compute the fraction of anonymized records
    that were uniquely re-identified.
    """
    return matches_df["anon_id"].nunique() / len(anon_df)
