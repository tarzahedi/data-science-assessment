import os

import pandas as pd
from prefect import flow, get_run_logger, task

from utilities.checks import (
    check_column_types,
    check_columns_exist,
    check_high_na,
    check_variability,
)
from utilities.data_utils import clean_column, generate_uuid
from utilities.file_handler import load_excel_file

SUPPLIER_1_PATH = os.path.join("./resources", "task_1", "supplier_data1.xlsx")
SUPPLIER_2_PATH = os.path.join("./resources", "task_1", "supplier_data2.xlsx")
EXPORT_PATH = os.path.join("./resources", "export", "inventory_dataset.csv")


@task
def load_data() -> tuple[pd.DataFrame]:
    """Load supplier Excel files into pandas DataFrames.

    I assume that we are using a fixed name file for each supplier.
    We can potentially generalize this even more and handle any kind of
    input data automatically. For simplicity, any preprocessing will happen
    considering the fixed format.

    We can also read data from any source other than excel.

    Returns:
        tuple[pd.DataFrame]: DataFrames for Supplier 1 and Supplier 2.
    """
    logger = get_run_logger()
    logger.info("Loading the data")

    logger.info(f"Loading supplier 1 file from {SUPPLIER_1_PATH}")
    supplier_1 = load_excel_file(SUPPLIER_1_PATH)
    logger.info(f"Loading supplier 2 file from {SUPPLIER_2_PATH}")
    supplier_2 = load_excel_file(SUPPLIER_2_PATH)

    logger.info("Applying data quality checks")
    supplier_data_quality_check(supplier_1=supplier_1, supplier_2=supplier_2)

    return supplier_1, supplier_2


@task
def supplier_data_quality_check(supplier_1, supplier_2) -> None:
    """Perform quality checks on supplier datasets.

    Checks include:
        - Required columns existence
        - High missing value columns
        - High variability in numeric columns
        - Duplicate Article IDs in Supplier 2

    Args:
        supplier_1 (pd.DataFrame): Supplier 1 DataFrame.
        supplier_2 (pd.DataFrame): Supplier 2 DataFrame.

    Raises:
        ValueError: If required columns are missing, columns exceed missing-value
                    threshold, or duplicate Article IDs are detected in Supplier 2.
    """

    logger = get_run_logger()

    # Expected columns for supplier 1
    SUPPLIER_1_COLUMNS = [
        "Quality/Choice",
        "Grade",
        "Finish",
        "Thickness (mm)",
        "Width (mm)",
        "Description",
        "Gross weight (kg)",
        "RP02",
        "RM",
        "Quantity",
        "AG",
        "AI",
    ]

    # Expected columns for supplier 2
    SUPPLIER_2_COLUMNS = [
        "Material",
        "Description",
        "Article ID",
        "Weight (kg)",
        "Quantity",
        "Reserved",
    ]

    # Expected types for supplier 1
    SUPPLIER_1_TYPES = {
        "Quality/Choice": str,
        "Grade": str,
        "Finish": str,
        "Thickness (mm)": float,
        "Width (mm)": int,
        "Description": str,
        "Gross weight (kg)": int,
        "RP02": float,
        "RM": float,
        "Quantity": float,
        "AG": float,
        "AI": float,
    }

    # Expected types for supplier 2
    SUPPLIER_2_TYPES = {
        "Material": str,
        "Description": str,
        "Article ID": int,
        "Weight (kg)": int,
        "Quantity": int,
        "Reserved": str,
    }

    # Check for column existence
    check_columns_exist(supplier_1, SUPPLIER_1_COLUMNS, "Supplier 1")
    check_columns_exist(supplier_2, SUPPLIER_2_COLUMNS, "Supplier 2")

    # Check for expected column types
    check_column_types(supplier_1, SUPPLIER_1_TYPES, "Supplier 1", strict=False)
    check_column_types(supplier_2, SUPPLIER_2_TYPES, "Supplier 2", strict=False)

    # Check for high null values in any column
    check_high_na(supplier_1, "Supplier 1", threshold=0.90)
    check_high_na(supplier_2, "Supplier 2", threshold=0.90)

    # Check for variability of values
    check_variability(supplier_1, "Supplier 1", 0.5)
    check_variability(supplier_2, "Supplier 2", 0.5)

    # check for duplicated IDs
    if supplier_2["Article ID"].duplicated().any():
        logger.error("[Supplier 2] Duplicate Article IDs detected")
        raise ValueError("[Supplier 2] Duplicate Article IDs detected")


@task
def preprocess_supplier_1(supplier_1: pd.DataFrame) -> pd.DataFrame:
    """Clean and preprocess Supplier 1 dataset.

    Tasks performed:
        - Standardize column names
        - Map description and finish values to consistent categories
        - Uppercase quality_choice
        - Rename columns for consistency
        - Add source column and article_id UUIDs
        - Drop unnecessary columns

    Args:
        supplier_1 (pd.DataFrame): Raw Supplier 1 DataFrame.

    Returns:
        pd.DataFrame: Preprocessed Supplier 1 DataFrame.
    """
    logger = get_run_logger()
    logger.info("Preprocessing supplier 1")
    # cleaning the columns
    supplier_1.columns = [clean_column(c) for c in supplier_1.columns]

    # Map description values
    description_mapping_1 = {
        "Längs- oder Querisse": "CRACKS",
        "Kantenfehler - FS-Kantenrisse": "EDGE CRACKS",
        "Sollmasse (Gewicht) unterschritten": "UNDERWEIGHT",
    }

    supplier_1["description"] = supplier_1["description"].apply(
        lambda x: description_mapping_1.get(x, x)
    )

    # Map finish values
    finish_mapping_1 = {
        "gebeizt und geglüht": "PICKLED & ANNEALED",  # 'PICKLED, ANNEALED'
        "ungebeizt": "UNPICKLED",
        "gebeizt": "PICKLED",
    }

    supplier_1["finish"] = supplier_1["finish"].apply(
        lambda x: finish_mapping_1.get(x, x)
    )

    # Uppercase column values for quality_choice:
    supplier_1["quality_choice"] = supplier_1["quality_choice"].str.upper()

    # Rename columns for consistency
    supplier_1.rename(
        columns={
            "grade": "grade_material",
            "gross_weight": "weight",
        },
        inplace=True,
    )

    # Adding source of the data
    supplier_1[["source"]] = "SUPPLIER_1"

    # Add missing article_id
    supplier_1["article_id"] = supplier_1.apply(lambda _: generate_uuid(), axis=1)

    # Select columns based on instruction -> remove: rp02, rm, ag, ai
    unwanted_columns_1 = ["rp02", "rm", "ag", "ai"]
    supplier_1 = supplier_1.drop(columns=unwanted_columns_1)

    return supplier_1


@task
def preprocess_supplier_2(supplier_2: pd.DataFrame) -> pd.DataFrame:
    """Clean and preprocess Supplier 2 dataset.

    Tasks performed:
        - Standardize column names
        - Map description values to finish categories
        - Rename columns for consistency
        - Add source column

    Args:
        supplier_2 (pd.DataFrame): Raw Supplier 2 DataFrame.

    Returns:
        pd.DataFrame: Preprocessed Supplier 2 DataFrame.
    """
    logger = get_run_logger()
    logger.info("Preprocessing supplier 2")
    # cleaning the columns
    supplier_2.columns = [clean_column(c) for c in supplier_2.columns]

    # Description looks like finish feature based on the values
    # Possible improvement: extract different features (e.g., finish, coating)
    # from description column
    description_mapping_2 = {
        "Material is Oiled": "OILED",
        "Material is Painted": "PAINTED",
        "Material is not Oiled": "NOT PAINTED",
    }
    supplier_2["description"] = supplier_2["description"].apply(
        lambda x: description_mapping_2.get(x, x)
    )

    # Rename columns for consistency
    supplier_2.rename(
        columns={"material": "grade_material", "description": "finish"}, inplace=True
    )

    # Adding source of the data
    supplier_2[["source"]] = "SUPPLIER_2"

    return supplier_2


@task
def merge_suppliers_data(
    supplier_1: pd.DataFrame, supplier_2: pd.DataFrame
) -> pd.DataFrame:
    """Merge preprocessed supplier datasets into a single inventory dataset.

    Missing values in key columns are filled with default values.

    I assume that we only have two suppliers. We can potentially merge
    any number of supplier data using input lists.

    Args:
        supplier_1 (pd.DataFrame): Preprocessed Supplier 1 DataFrame.
        supplier_2 (pd.DataFrame): Preprocessed Supplier 2 DataFrame.

    Returns:
        pd.DataFrame: Combined inventory dataset.
    """
    logger = get_run_logger()
    logger.info("Merging supplier data...")
    inventory_dataset = pd.concat([supplier_1, supplier_2], ignore_index=True)

    # handling missing values
    inventory_dataset[["quality_choice", "reserved"]] = inventory_dataset[
        ["quality_choice", "reserved"]
    ].fillna("UNKNOWN")

    # # Optional for machine learning purpose
    # inventory_dataset[["thickness", "width"]].fillna(inventory_dataset[["thickness_mm", "width_mm"]].mean())

    return inventory_dataset


@task
def export_inventory_dataset(inventory_dataset: pd.DataFrame, export_path: str) -> None:
    """Export the final inventory dataset to CSV.

    Args:
        inventory_dataset (pd.DataFrame): The dataset to export.
        export_path (str): File path for the exported CSV.
    """
    logger = get_run_logger()
    inventory_dataset.to_csv(export_path, index=False)
    logger.info(f"Inventory dataset is written successfully in {export_path}")


@task
def preprocess_suppliers(
    supplier_1: pd.DataFrame, supplier_2: pd.DataFrame
) -> pd.DataFrame:
    """Load, quality-check, and preprocess supplier datasets.

    I assume that we have two types of Suppliers and we will preprocess those.

    Returns:
        tuple[pd.DataFrame, pd.DataFrame]: Preprocessed DataFrames for Supplier 1
        and Supplier 2.
    """
    logger = get_run_logger()

    logger.info("Preprocessing suppliers")
    supplier_1_cleaned = preprocess_supplier_1(supplier_1)
    supplier_2_cleaned = preprocess_supplier_2(supplier_2)

    return supplier_1_cleaned, supplier_2_cleaned


@flow(name="task_1_pipeline")
def pipeline() -> None:
    """Main ETL pipeline flow for Task 1.

    Steps:
        1. Load data for suppliers
        2. Preprocess supplier files
        2. Merge supplier files
        3. Export merged inventory dataset
    """

    # 1. Extract data
    supplier_1, supplier_2 = load_data()

    # 2. Preprocess supplier files
    supplier_1, supplier_2 = preprocess_suppliers(supplier_1, supplier_2)

    # 3. Merge supplier files
    inventory_dataset = merge_suppliers_data(supplier_1, supplier_2)

    # 4. Export inventory dataset
    export_inventory_dataset(inventory_dataset, EXPORT_PATH)
