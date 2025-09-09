import numpy as np
import pandas as pd
from prefect import get_run_logger


def interval_matrix(mins: np.ndarray, maxs: np.ndarray, sim_method: str = "iou"):
    """
    Compute pairwise similarity between intervals using IoU or overlap ratio.

    Args:
        mins (np.ndarray): Array of shape (n_samples, n_dims),
            minimum values per interval.
        maxs (np.ndarray): Array of shape (n_samples, n_dims),
            maximum values per interval.
        sim_method (str, optional): Similarity metric.
            - "iou": Intersection over Union.
            - "overlap": Intersection over the smaller interval.
            Defaults to "iou".

    Returns:
        np.ndarray: Similarity matrix of shape (n_samples, n_samples),
        where each entry is the similarity between two intervals.
    """
    n, m = mins.shape

    # Expand dimensions to (n, 1, m) and (1, n, m) to broadcast pairwise
    mins_exp = mins[:, np.newaxis, :]  # shape (n,1,m)
    maxs_exp = maxs[:, np.newaxis, :]  # shape (n,1,m)

    mins_pair = mins[np.newaxis, :, :]  # shape (1,n,m)
    maxs_pair = maxs[np.newaxis, :, :]  # shape (1,n,m)

    # Compute intersection and union
    intersection = np.maximum(
        0, np.minimum(maxs_exp, maxs_pair) - np.maximum(mins_exp, mins_pair)
    )

    if sim_method == "iou":
        union = np.maximum(maxs_exp, maxs_pair) - np.minimum(mins_exp, mins_pair)
        iou = np.divide(
            intersection, union, out=np.zeros_like(intersection), where=union != 0
        )
    elif sim_method == "overlap":
        lengths = np.minimum(maxs_exp - mins_exp, maxs_pair - mins_pair)
        iou = np.divide(
            intersection, lengths, out=np.zeros_like(intersection), where=lengths != 0
        )
    else:
        raise ValueError("sim_method must be 'iou' or 'overlap'")

    # Aggregate across dimensions (average)
    sims = np.nanmean(iou, axis=2)

    return sims


def dim_similarity(df: pd.DataFrame, sim_method: str = "iou") -> np.ndarray:
    """
    Compute pairwise similarity scores for dimensional features.

    Dimensions considered:
        thickness, width, length, height, weight,
        inner diameter, outer diameter.

    Args:
        df (pd.DataFrame): Input dataframe containing min/max columns for dimensions.
        sim_method (str, optional): Interval similarity method ("iou" or "overlap").
            Defaults to "iou".

    Returns:
        np.ndarray: Square similarity matrix where entry (i, j) is
        the similarity between item i and item j based on dimensions.
    """
    dimension_cols = [
        ("thickness_min", "thickness_max"),
        ("width_min", "width_max"),
        ("length_min", "length_max"),
        ("height_min", "height_max"),
        ("weight_min", "weight_max"),
        ("inner_diameter_min", "inner_diameter_max"),
        ("outer_diameter_min", "outer_diameter_max"),
    ]

    existing_dims = []
    for min_col, max_col in dimension_cols:
        if min_col in df.columns:
            if max_col is None or max_col not in df.columns:
                max_col = min_col
            df[min_col] = df[min_col].fillna(df[max_col])
            df[max_col] = df[max_col].fillna(df[min_col])
            existing_dims.append((min_col, max_col))

    dim_min = df[[c[0] for c in existing_dims]].to_numpy()
    dim_max = df[[c[1] for c in existing_dims]].to_numpy()

    dim_sims = interval_matrix(dim_min, dim_max, sim_method=sim_method)
    # Remove self matches
    np.fill_diagonal(dim_sims, 0)

    return dim_sims


def cat_similarity(df: pd.DataFrame, sim_method: str = "partial") -> np.ndarray:
    """
    Compute pairwise similarity scores for categorical features.

    Categorical columns considered:
        coating, finish, form, surface_type.

    Args:
        df (pd.DataFrame): Input dataframe with categorical columns.
        sim_method (str, optional): Method for categorical comparison.
            - "partial": Fraction of matching categories.
            - "exact": 1 if all categories match, else 0.
            - "jaccard": Jaccard index based on one-hot encoding.
            Defaults to "partial".

    Raises:
        ValueError: If sim_method is not supported.

    Returns:
        np.ndarray: Square similarity matrix where entry (i, j) is
        the similarity between item i and item j based on categorical attributes.
    """
    categorical_cols = ["coating", "finish", "form", "surface_type"]
    categorical_cols = [c for c in categorical_cols if c in df.columns]
    cat_array = (
        df[categorical_cols].to_numpy(dtype=object) if categorical_cols else None
    )
    cat_array = np.array([tuple(row) for row in cat_array])
    cat_match = cat_array[:, None] == cat_array[None, :]
    # Vectorized pairwise equality
    if sim_method == "partial":
        cat_sims = cat_match.mean(-1).astype(float)  # Mean of all categories that match
    elif sim_method == "exact":
        cat_sims = cat_match.all(-1).astype(int)  # All categories match
    elif sim_method == "jaccard":
        # Vectorized Jaccard for categorical tuples
        # Convert to 2D one-hot encoding per column
        # First, find unique values per column
        n_rows, n_cols = cat_array.shape
        encoded = []
        for c in range(n_cols):
            uniques, inv = np.unique(cat_array[:, c], return_inverse=True)
            one_hot = np.zeros((n_rows, len(uniques)), dtype=bool)
            one_hot[np.arange(n_rows), inv] = True
            encoded.append(one_hot)
        # Concatenate all columns
        encoded = np.hstack(encoded)  # shape: (n_rows, total_unique_categories)

        # Compute pairwise intersection and union
        intersection = np.dot(encoded, encoded.T)
        row_sums = encoded.sum(axis=1)
        union = row_sums[:, None] + row_sums[None, :] - intersection

        cat_sims = intersection / np.where(union == 0, 1, union)
    else:
        raise ValueError(f"Similarity method {sim_method} is not supported!")

    # Remove self matches
    np.fill_diagonal(cat_sims, 0)

    return cat_sims


def cosine_sim_matrix(grades: np.ndarray) -> np.ndarray:
    """
    Compute pairwise cosine similarity between rows of grade features.

    NaN values are ignored by restricting comparisons to non-missing features.

    Args:
        grades (np.ndarray): 2D array of numeric features with possible NaNs.

    Returns:
        np.ndarray: Square cosine similarity matrix.
    """
    n, m = grades.shape
    sims = np.zeros((n, n))

    for i in range(n):
        row_i = grades[i]
        valid_mask = ~np.isnan(row_i)  # which features are valid for this row
        vecs = grades[:, valid_mask]  # pick only valid features
        row_i_vec = row_i[valid_mask]

        # Compute norms
        norms = np.linalg.norm(vecs, axis=1) * np.linalg.norm(row_i_vec)
        # Avoid division by zero
        norms[norms == 0] = 1e-10

        # Compute cosine similarity
        sims[i] = np.dot(vecs, row_i_vec) / norms

    return sims


def grade_similarity(df: pd.DataFrame, sim_method: str = "cosine") -> np.ndarray:
    """
    Compute pairwise similarity scores for grade-level numeric properties.

    Supported features (if present in dataframe):
        yield strength, tensile strength, elongation,
        carbon, manganese, silicon, sulfur, phosphorus, aluminum.

    Args:
        df (pd.DataFrame): Input dataframe with grade property ranges.
        sim_method (str, optional): Similarity method. Currently supports:
            - "cosine" (default).

    Raises:
        ValueError: If sim_method is not supported.

    Returns:
        np.ndarray: Square similarity matrix for grade features.
    """
    numeric_range_cols = [
        ("yield_strength_min", "yield_strength_max"),
        ("tensile_strength_min", "tensile_strength_max"),
        ("elongation_min", "elongation_max"),
        ("carbon_min", "carbon_max"),
        ("manganese_min", "manganese_max"),
        ("silicon_min", "silicon_max"),
        ("sulfur_min", "sulfur_max"),
        ("phosphorus_min", "phosphorus_max"),
        ("aluminum_min", "aluminum_max"),
    ]

    mid_cols = []
    for min_col, max_col in numeric_range_cols:
        if min_col in df.columns and max_col in df.columns:
            mid_col = min_col.replace("_min", "_mid")
            df[mid_col] = df[[min_col, max_col]].mean(axis=1, skipna=True)
            mid_cols.append(mid_col)

    grade_array = df[mid_cols].to_numpy(dtype=float) if mid_cols else None

    similarity_func = cosine_sim_matrix
    if sim_method == "cosine":
        similarity_func = cosine_sim_matrix
    # elif sim_method == "":
    else:
        raise ValueError(f"Similarity method {sim_method} is not supported!")
    grade_sims = (
        similarity_func(grade_array)
        if grade_array is not None
        else np.zeros((len(df), len(df)))
    )
    np.fill_diagonal(grade_sims, 0)
    return grade_sims


def aggregate_sim_score(
    df: pd.DataFrame,
    w_dim: float = 0.3,
    w_cat: float = 0.3,
    w_grade: float = 0.4,
    dimensional_method: str = "iou",
    categorical_method: str = "partial",
    grade_method: str = "cosine",
) -> np.ndarray:
    """
    Aggregate similarity scores across dimensions, categorical, and grade features.

    Args:
        df (pd.DataFrame): Input dataset containing required features.
        w_dim (float, optional): Weight for dimensional similarity. Defaults to 0.3.
        w_cat (float, optional): Weight for categorical similarity. Defaults to 0.3.
        w_grade (float, optional): Weight for grade similarity. Defaults to 0.4.
        dimensional_method (str, optional): Method for dimension comparison.
            Defaults to "iou".
        categorical_method (str, optional): Method for categorical comparison.
            Defaults to "partial".
        grade_method (str, optional): Method for grade comparison. Defaults to "cosine".

    Returns:
        np.ndarray: Weighted aggregate similarity matrix.
    """
    dim_sims = dim_similarity(df, sim_method=dimensional_method)
    cat_sims = cat_similarity(df, sim_method=categorical_method)
    grade_sims = grade_similarity(df, sim_method=grade_method)

    aggregate_sims = w_dim * dim_sims + w_cat * cat_sims + w_grade * grade_sims

    np.fill_diagonal(aggregate_sims, 0)

    return aggregate_sims


def generate_top_k(
    df: pd.DataFrame,
    w_dim: float = 0.3,
    w_cat: float = 0.3,
    w_grade: float = 0.4,
    top_k: int = 3,
    dimensional_method: str = "iou",
    categorical_method: str = "partial",
    grade_method: str = "cosine",
) -> pd.DataFrame:
    """
    Generate top-k most similar matches for each item in the dataset.

    Args:
        df (pd.DataFrame): Input dataset containing id and feature columns.
        w_dim (float, optional): Weight for dimensional similarity. Defaults to 0.3.
        w_cat (float, optional): Weight for categorical similarity. Defaults to 0.3.
        w_grade (float, optional): Weight for grade similarity. Defaults to 0.4.
        top_k (int, optional): Number of top matches to return for each item.
            Defaults to 3.
        dimensional_method (str, optional): Method for dimension comparison.
            Defaults to "iou".
        categorical_method (str, optional): Method for categorical comparison.
            Defaults to "partial".
        grade_method (str, optional): Method for grade comparison. Defaults to "cosine".

    Returns:
        pd.DataFrame: DataFrame with columns:
            - rfq_id: ID of the query item.
            - match_id: ID of the matched item.
            - similarity_score: Score of the match.
    """
    logger = get_run_logger()
    logger.info(f"Generating top k={top_k} results")
    records = []
    aggregate_sims = aggregate_sim_score(
        df,
        w_dim,
        w_cat,
        w_grade,
        dimensional_method=dimensional_method,
        categorical_method=categorical_method,
        grade_method=grade_method,
    )
    for i, row in enumerate(aggregate_sims):
        top_idx = np.argsort(row)[-top_k:][::-1]  # best → worst
        for j in top_idx:
            records.append(
                {
                    "rfq_id": df.loc[i, "id"],
                    "match_id": df.loc[j, "id"],
                    "similarity_score": float(row[j]),  # force float, no NaN
                }
            )

    similarity_df = pd.DataFrame(records)

    return similarity_df
