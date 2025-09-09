import pandas as pd
from prefect import get_run_logger


def check_columns_exist(df: pd.DataFrame, expected_cols: list[str], table_name: str):
    """Check that a DataFrame contains all expected columns.

    Args:
        df (pd.DataFrame): The DataFrame to validate.
        expected_cols (list[str]): A list of column names that must be present.
        table_name (str): Name of the table (used for error messages).

    Raises:
        ValueError: If any expected column is missing from the DataFrame.
    """
    missing = [col for col in expected_cols if col not in df.columns]
    if missing:
        raise ValueError(f"{table_name}: Missing required columns: {missing}")


def check_column_types(
    df: pd.DataFrame, type_mapping: dict[str, type], table_name: str, strict=False
) -> None:
    """Check if the DataFrame columns match the expected data types.

    Args:
        df (pd.DataFrame): The DataFrame to validate.
        type_mapping (dict[str, type]): Mapping of column names to expected Python types (e.g., int, float, str).
        table_name (str): Name of the table (used in error messages).
        strict (bool): Whether to strictly check for type mismatches or only warn

    Raises:
        TypeError: If any column has a type that does not match the expected type.
    """
    logger = get_run_logger()
    mismatches = []

    for col, expected_type in type_mapping.items():
        actual_type = (
            df[col].dropna().map(type).mode()[0]
            if not df[col].dropna().empty
            else expected_type
        )
        if actual_type is not expected_type:
            mismatches.append((col, expected_type.__name__, actual_type.__name__))

    if mismatches:
        mismatch_str = ", ".join(
            [f"{col} (expected: {exp}, actual: {act})" for col, exp, act in mismatches]
        )
        msg = f"{table_name}: Column type mismatches detected: {mismatch_str}"
        if strict:
            raise TypeError(msg)
        else:
            logger.warning(msg)
    else:
        logger.info(f"{table_name}: All column types match expected types.")


def check_high_na(
    df: pd.DataFrame,
    table_name: str,
    threshold: float = 0.9,
    should_raise: bool = True,
):
    """Check for columns in a DataFrame with a high proportion of missing values.

    Args:
        df (pd.DataFrame): The DataFrame to validate.
        table_name (str): Name of the table (used for error/warning messages).
        threshold (float, optional): The proportion of missing values above which
            a column is flagged. Defaults to 0.9.
        should_raise (bool, optional): Whether to raise an error if issues are
            found. If False, logs a warning instead. Defaults to True.

    Raises:
        ValueError: If `should_raise` is True and one or more columns exceed
            the missing-value threshold.
    """
    logger = get_run_logger()

    issues = []
    for col in df.columns:
        na_ratio = df[col].isna().mean()
        if na_ratio >= threshold:
            issues.append((col, f"{na_ratio:.1%} missing"))
    if issues:
        if should_raise:
            raise ValueError(
                f"{table_name}: Columns with ≥{int(threshold*100)}%"
                f" missing values: {issues}"
            )
        else:
            logger.warning(
                f"{table_name}: Columns with ≥{int(threshold*100)}%"
                f" missing values: {issues}"
            )


def check_variability(
    df: pd.DataFrame, table_name: str, cv_threshold: float = 1.0
) -> None:
    """ "Check for high variability in numeric columns of a DataFrame using the
    coefficient of variation (CV = standard deviation / mean).

    Args:
        df (pd.DataFrame): The DataFrame to validate.
        table_name (str): Name of the table (used for logging).
        cv_threshold (float, optional): The CV threshold above which columns
            are considered to have high variability. Defaults to 1.0.

    Notes:
        Logs a warning for each column exceeding the threshold. Does not raise errors.
    """
    logger = get_run_logger()
    numeric_cols = df.select_dtypes(include="number").columns
    for col in numeric_cols:
        std = df[col].std()
        mean = df[col].mean()
        if mean != 0:
            cv = std / mean
            if cv > cv_threshold:
                logger.warning(
                    f"{table_name}: Column '{col}' has high variability "
                    f"(std={std:.2f}, mean={mean:.2f}, CV={cv:.2f})"
                )
