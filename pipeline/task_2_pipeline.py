import os

import pandas as pd
from prefect import flow, get_run_logger, task

from utilities.checks import check_columns_exist, check_high_na, check_variability
from utilities.data_utils import (
    char_finder,
    clean_column,
    custom_fillna,
    drop_high_missing,
    fill_rfq_with_ref,
    normalize_grade,
    parse_columns,
    suffix_finder,
)
from utilities.file_handler import load_csv_file
from utilities.similarity import generate_top_k

RFQ_FILE_PATH = os.path.join("./resources", "task_2", "rfq.csv")
REFERENCE_FILE_PATH = os.path.join("./resources", "task_2", "reference_properties.tsv")
EXPORT_PATH = os.path.join("./resources", "export", "top3.csv")


@task
def load_data() -> tuple[pd.DataFrame]:
    """Load RFQ and reference property data from disk and validate quality.

    Returns:
        tuple[pd.DataFrame]:
            - rfq (pd.DataFrame): Request for Quotation dataset.
            - ref (pd.DataFrame): Reference properties dataset.
    """
    logger = get_run_logger()
    logger.info("Loading data")
    rfq = load_csv_file(RFQ_FILE_PATH)
    ref = load_csv_file(REFERENCE_FILE_PATH, sep="\t")

    data_quality_check(rfq, ref)

    return rfq, ref


@task
def data_quality_check(rfq: pd.DataFrame, ref: pd.DataFrame) -> None:
    """Validate RFQ and reference datasets for structure and data quality.

    Checks include:
      - Column existence.
      - High missing-value ratios.
      - Variability of numerical features.

    Args:
        rfq (pd.DataFrame): RFQ dataset.
        ref (pd.DataFrame): Reference properties dataset.
    """
    logger = get_run_logger()
    logger.info("Applying data quality checks")

    # Expected columns for rfq
    RFQ_COLUMNS = [
        "id",
        "grade",
        "grade_suffix",
        "coating",
        "finish",
        "surface_type",
        "surface_protection",
        "form",
        "thickness_min",
        "thickness_max",
        "width_min",
        "width_max",
        "length_min",
        "height_min",
        "height_max",
        "weight_min",
        "weight_max",
        "inner_diameter_min",
        "inner_diameter_max",
        "outer_diameter_min",
        "outer_diameter_max",
        "yield_strength_min",
        "yield_strength_max",
        "tensile_strength_min",
        "tensile_strength_max",
    ]

    # Expected columns for references
    REF_COLUMNS = [
        "Grade/Material",
        "UNS_No",
        "Steel_No",
        "Standards",
        "Carbon (C)",
        "Manganese (Mn)",
        "Silicon (Si)",
        "Sulfur (S)",
        "Phosphorus (P)",
        "Chromium (Cr)",
        "Nickel (Ni)",
        "Molybdenum (Mo)",
        "Vanadium (V)",
        "Tungsten (W)",
        "Cobalt (Co)",
        "Copper (Cu)",
        "Aluminum (Al)",
        "Titanium (Ti)",
        "Niobium (Nb)",
        "Boron (B)",
        "Nitrogen (N)",
        "Tensile strength (Rm)",
        "Yield strength (Re or Rp0.2)",
        "Elongation (A%)",
        "Reduction of area (Z%)",
        "Hardness (HB, HV, HRC)",
        "Impact toughness (Charpy V-notch)",
        "Fatigue limit",
        "Creep resistance",
        "Source_Pages",
        "Application",
        "Category",
        "Nb + V + Ti (Others)",
        "Coating",
    ]

    # Check for column existence
    check_columns_exist(rfq, RFQ_COLUMNS, "Rfq")
    check_columns_exist(ref, REF_COLUMNS, "References")

    # Check for high null values in any column
    check_high_na(rfq, "Rfq", threshold=0.90, should_raise=False)
    check_high_na(ref, "References", threshold=0.90, should_raise=False)

    # Check for variability of values
    check_variability(rfq, "Rfq", 0.5)
    check_variability(ref, "References", 0.5)


@task
def preprocess_data(rfq: pd.DataFrame, ref: pd.DataFrame) -> tuple[pd.DataFrame]:
    """Preprocess RFQ and reference datasets for normalization and alignment.

    Steps include:
      - Cleaning and normalizing column names.
      - Standardizing grade values and finishes.
      - Handling categorical column casing.
      - Extracting suffixes from grades and imputing finish values.
      - Parsing reference columns containing ranges or special characters.

    Args:
        rfq (pd.DataFrame): Raw RFQ dataset.
        ref (pd.DataFrame): Raw reference dataset.

    Returns:
        tuple[pd.DataFrame]:
            - Cleaned RFQ dataset.
            - Cleaned reference dataset.
    """
    logger = get_run_logger()
    logger.info("Preprocessing the data")
    # TODO: Extract more code from this function into individual functions
    finish_dict = {
        "+AR": "AS ROLLED (+AR)",
        "OILED": "OILED (O)",
        "LIGHTLY OILED": "LIGHTLY OILED (L)",
        "+N": "NORMALIZED (+N)",
        "COLD ROLLED": "COLD ROLLED (+CR)",
        "+A": "SOFT ANNEALED (+A)",
    }

    suffix_finish = {
        "+N": "Normalized (+N)",
        "+C": "Cold Drawn (+C)",
        "QT": "Quenched and Tempered (+QT)",
    }

    logger = get_run_logger()
    logger.info("Preprocess data")
    # Clean up column names
    ref.columns = [clean_column(c) for c in ref.columns]
    ref.rename(columns={"grade_material": "grade"}, inplace=True)  # Keep consistency

    rfq.columns = [clean_column(c) for c in rfq.columns]

    # Normalize grade values
    rfq.loc[rfq["grade"].notna(), "grade"] = rfq.loc[
        rfq["grade"].notna(), "grade"
    ].apply(normalize_grade)

    ref["grade"] = ref["grade"].apply(normalize_grade)

    # Remove potential duplicates in Grade
    ref = ref.drop_duplicates(subset=["grade"]).reset_index(drop=True)

    #
    rfq_grade_list = list(rfq.grade.unique())
    ref_grade_list = list(ref.grade.unique())
    # Check all rfq grade exist in ref
    missing = [item for item in rfq_grade_list if item not in ref_grade_list]
    if len(missing) > 1:
        logger.warning("Missing rfq grades in the Reference: {missing}")

    # Normalize categorical columns
    rfq_categorical_columns = (
        rfq.select_dtypes(include="object").drop(columns=["id"]).columns
    )
    rfq[rfq_categorical_columns] = rfq[rfq_categorical_columns].apply(
        lambda x: x.str.upper()
    )

    ref_categorical_columns = ref.select_dtypes(include="object").columns
    ref[ref_categorical_columns] = ref[ref_categorical_columns].apply(
        lambda x: x.str.upper()
    )

    # Handle finish column, normalize data
    rfq["finish"] = rfq["finish"].apply(lambda x: finish_dict.get(x, x))

    # Impute grade_suffix by extracting suffix from grade
    rfq["grade_suffix"] = rfq["grade"].apply(suffix_finder)

    # Impute finish by grade_suffix
    rfq.loc[rfq["finish"].isna(), "finish"] = rfq[rfq["finish"].isna()][
        "grade_suffix"
    ].apply(lambda x: suffix_finish.get(x, x))

    # Handling special characters in column names
    cols_with_special = char_finder(ref)
    logger.info(
        (
            "Columns renamed that contained special characters: "
            f"{', '.join(cols_with_special)}"
        )
    )

    # Parse reference columns with special characters
    # Handling ranges and missing values
    ref = parse_columns(ref, char_finder(ref))

    return rfq, ref


@task
def merge_rfq_reference(rfq: pd.DataFrame, ref: pd.DataFrame) -> pd.DataFrame:
    """Join RFQ dataset with reference dataset and enrich with missing values.

    Steps include:
      - Merging on grade.
      - Imputing missing RFQ values from reference.
      - Dropping high-missing columns.
      - Applying custom fill strategies for residual NaNs.

    Args:
        rfq (pd.DataFrame): Cleaned RFQ dataset.
        ref (pd.DataFrame): Cleaned reference dataset.

    Returns:
        pd.DataFrame: Enriched RFQ dataset.
    """
    logger = get_run_logger()
    logger.info("Enriching rfq table with references")
    merge_df = rfq.merge(ref, on="grade", how="left", suffixes=("_rfq", "_ref"))

    # Using reference columns to impute rfq data
    merge_df = fill_rfq_with_ref(merge_df)

    # Droping columns with high missing values
    merge_df = drop_high_missing(merge_df, threshold=0.9)

    # Handling additional missing values (impute/etc)
    rfq_enriched = custom_fillna(merge_df).copy()

    return rfq_enriched


@task
def calculate_similarity(enriched_rfq: pd.DataFrame) -> pd.DataFrame:
    """Compute similarity scores between RFQ entries.

    Uses weighted similarity across:
      - Dimensional properties (IoU).
      - Categorical properties (partial string matching).
      - Grade-level properties (cosine similarity).

    Args:
        enriched_rfq (pd.DataFrame): Enriched RFQ dataset with reference features.

    Returns:
        pd.DataFrame: Similarity table containing top-k matches.
    """
    logger = get_run_logger()
    logger.info("Calculating similarity")
    w_dim = 0.3
    w_cat = 0.3
    w_grade = 0.4
    top_k = 3
    dimensional_method = "iou"
    categorical_method = "partial"
    grade_method = "cosine"
    logger.info(f"Using w_dim={w_dim}, w_cat={w_cat}, w_grade={w_grade}")
    logger.info(
        (
            f"Using dimensional_method={dimensional_method}, "
            f"categorical_method={categorical_method}, "
            f"grade_method={grade_method}"
        )
    )
    similarity_dataframe = generate_top_k(
        enriched_rfq,
        w_dim=w_dim,
        w_cat=w_cat,
        w_grade=w_grade,
        dimensional_method=dimensional_method,
        categorical_method=categorical_method,
        grade_method=grade_method,
        top_k=top_k,
    )

    return similarity_dataframe


@task
def export_top_similarity_dataset(
    similarity_table: pd.DataFrame, export_path: str
) -> None:
    """Export the top similarity results to CSV.

    Args:
        similarity_table (pd.DataFrame): Similarity results DataFrame.
        export_path (str): Output file path for the CSV.
    """
    logger = get_run_logger()
    similarity_table.to_csv(export_path, index=False)
    logger.info(f"top_3 dataset is written successfully in {export_path}")


@flow(name="task_2_pipeline")
def pipeline() -> None:
    """ETL pipeline for RFQ similarity matching.

    Steps:
      1. Load RFQ and reference datasets.
      2. Preprocess and normalize datasets.
      3. Merge RFQ with reference data.
      4. Calculate similarity between entries.
      5. Export top-k similarity matches to CSV.
    """

    # 1. Load data
    rfq, ref = load_data()

    # 2. preprocess_data
    rfq_cleaned, ref_cleaned = preprocess_data(rfq, ref)

    # 3. Merge (join) rfq and reference
    rfq_enriched = merge_rfq_reference(rfq_cleaned, ref_cleaned)

    # 4. Calculate similarity
    similarity_table = calculate_similarity(rfq_enriched)

    # 5. Export data
    export_top_similarity_dataset(similarity_table, EXPORT_PATH)
