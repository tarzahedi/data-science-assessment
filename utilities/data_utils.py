import uuid

import numpy as np
import pandas as pd


def clean_column(name: str) -> str:
    """Standardize a column name by converting it to lowercase, replacing
    special characters with underscores, collapsing multiple underscores,
    and stripping underscores from the start and end.

    Args:
        name (str): The original column name.

    Returns:
        str: A cleaned and standardized column name.
    """
    # Lowercase
    name = name.lower()
    if "(" in name and ")" in name:
        start = name.index("(")
        end = name.index(")")
        name = (name[:start] + name[end + 1 :]).strip()
    # Replace common separators with underscore
    for ch in [" ", "/", "-", ".", "+"]:
        name = name.replace(ch, "_")
    # Remove double underscores
    while "__" in name:
        name = name.replace("__", "_")
    # Strip underscores at ends
    name = name.strip("_")
    return name


def generate_uuid() -> str:
    """Generate a new universally unique identifier (UUID).

    Returns:
        str: A string representation of a randomly generated UUID4.
    """
    return str(uuid.uuid4())


def normalize_grade(grade: str) -> str:
    """Normalize a steel grade string by uppercasing and removing spaces.

    Args:
        grade (str): Original grade string.

    Returns:
        str: Normalized grade string.
    """
    # Uppercase and remove spaces
    grade_norm = str(grade).upper().replace(" ", "")
    return grade_norm


def suffix_finder(grade: pd.DataFrame) -> pd.Series:
    """Extract grade suffix if present (e.g., +N, -AR).

    A suffix is identified as the substring starting with '+' or '-'.
    If the suffix is longer than 4 characters, it is considered invalid
    and None is returned.

    Args:
        grade (pd.DataFrame): A grade string.

    Returns:
        pd.Series: Extracted suffix or None if not found.
    """
    # Initialize suffix
    grade_suffix = None
    # Look for suffix starting with '+' or '-'
    for i, c in enumerate(str(grade)):
        if c in "+-":
            grade_suffix = grade[i:]
            break
    if (grade_suffix is not None) and (len(grade_suffix) > 4):
        # Handing non suffix cases where the grade name is long and contain "-"
        return None
    return grade_suffix


def parse_range(s: str) -> tuple[float | None, float | None]:
    """Parse a string representing a numeric range or inequality.

    Handles:
        - Ranges: "5-10" → (5.0, 10.0)
        - Greater/less than: "≥5" → (5.0, None), "<3" → (0.0, 3.0)
        - Percentages: ">5%" → (5.0, 100.0)
        - Single numbers: "7" → (7.0, 7.0)

    Args:
        s (str): String value to parse.

    Returns:
        tuple[float | None, float | None]: Minimum and maximum values, or (None, None).
    """
    if s is None or str(s).strip() == "":
        return None, None

    s = str(s).strip()
    allowed = set("0123456789.-≤≥<>% ")

    # Remove trailing units (keep only allowed characters)
    i = len(s)
    while i > 0 and s[i - 1] not in allowed:
        i -= 1
    s = s[:i].strip()

    # Normalize characters
    s = s.replace("–", "-").replace(" ", "")

    # --- Case 1: range with dash ("5-10") ---
    if "-" in s and not s.startswith("-"):
        try:
            a, b = map(float, s.split("-", 1))
            return min(a, b), max(a, b)
        except ValueError:
            return None, None

    # --- Case 2: inequalities ---
    if s.startswith(("≥", ">")):
        if s.endswith("%"):
            try:
                n = float(s[1:-1])
                return n, 100.0
            except ValueError:
                return None, None
        else:
            try:
                n = float(s[1:])
                return n, None
            except ValueError:
                return None, None

    if s.startswith(("≤", "<")):
        if s.endswith("%"):
            try:
                n = float(s[1:-1])
                return 0.0, n
            except ValueError:
                return None, None
        else:
            try:
                n = float(s[1:])
                return 0.0, n
            except ValueError:
                return None, None

    # --- Case 3: single number ---
    try:
        n = float(s)
        return n, n
    except ValueError:
        return None, None


def char_finder(df: pd.DataFrame) -> list[str]:
    """Identify columns containing special characters in their values.

    Special characters checked: "-", "≤", "≥", "<", ">", "%".

    Args:
        df (pd.DataFrame): Input dataframe.

    Returns:
        list[str]: List of column names containing special characters.
    """
    # Look for column with these character
    special_chars = ["-", "≤", "≥", "<", ">", "%"]

    cols_with_special = []

    for col in df.columns:
        if df[col].dtype == "object":  # only check text columns
            # Check if any value in the column contains one of the special characters
            if (
                df[col]
                .astype(str)
                .str.contains("|".join(special_chars), na=False)
                .any()
            ):
                cols_with_special.append(col)
    return cols_with_special


def parse_columns(
    df: pd.DataFrame, cols_with_special: list[str], threshold: int = 70
) -> pd.DataFrame:
    """Parse and split columns with range-like strings into min/max numeric columns.

    Steps:
        - Compute missing value percentage per column.
        - Drop columns above the missing threshold.
        - Parse ranges and inequalities into numeric min/max values.
        - Replace original columns with parsed min/max pairs.

    Args:
        df (pd.DataFrame): Input DataFrame.
        cols_with_special (list[str]): List of columns to parse.
        threshold (int, optional): Maximum allowed % of missing values. Defaults to 70.

    Returns:
        pd.DataFrame: DataFrame with parsed min/max columns.
    """
    # Compute missing percentage, excluding 'grade' and 'application'
    missing_pct = (
        df[cols_with_special]
        .drop(columns=["grade", "application"], errors="ignore")
        .isna()
        .mean()
        * 100
    )

    # Filter columns below threshold
    cols_below_threshold = missing_pct[missing_pct < threshold].index
    # Parse ranges and create new columns
    for col in cols_below_threshold:
        df[[f"{col}_min", f"{col}_max"]] = df[col].apply(parse_range).apply(pd.Series)
        df.drop(columns=[col], inplace=True)

    return df


def fill_rfq_with_ref(df: pd.DataFrame) -> pd.DataFrame:
    """
    Fills NaN values in '_rfq' columns using corresponding '_ref' columns.

    Parameters:
        df (pd.DataFrame): The DataFrame containing '_ref' and '_rfq' columns.

    Returns:
        pd.DataFrame: DataFrame with NaNs in '_rfq' columns filled.
    """
    # Loop over columns ending with '_rfq'
    for col in df.columns:
        if col.endswith("_rfq"):
            col_ref = col.replace("_rfq", "_ref")
            if col_ref in df.columns:
                df[col] = df[col].fillna(df[col_ref])
    df = df.drop(columns=[col for col in df.columns if col.endswith("_ref")])
    df = df.rename(
        columns={
            col: col.replace("_rfq", "") for col in df.columns if col.endswith("_rfq")
        }
    )

    return df


def drop_high_missing(df: pd.DataFrame, threshold: float = 0.7) -> pd.DataFrame:
    """
    Drop columns with more than 'threshold' proportion of missing values.

    Parameters:
        df (pd.DataFrame): Input dataframe
        threshold (float): Proportion of missing values to tolerate (default=0.9)

    Returns:
        pd.DataFrame: DataFrame with columns dropped
    """
    missing_ratio = df.isnull().mean()  # proportion of missing values per column
    cols_to_drop = missing_ratio[missing_ratio > threshold].index
    return df.drop(columns=cols_to_drop)


def custom_fillna(df: pd.DataFrame) -> pd.DataFrame:
    """Fill missing values with appropriate strategies based on column type.

    Strategies:
        - Object/categorical: Replace with "UNKNOWN".
        - Columns ending with "_min": Fill with column minimum.
        - Columns ending with "_max": Fill with column maximum.
        - Other numeric columns: Fill with column mean.

    Args:
        df (pd.DataFrame): Input DataFrame.

    Returns:
        pd.DataFrame: DataFrame with missing values imputed.
    """
    df = df.copy()

    for col in df.columns:
        if df[col].dtype == "object":
            # categorical/string
            df[col] = df[col].fillna("UNKNOWN")
        elif col.endswith("_min"):
            df[col] = df[col].fillna(df[col].min(skipna=True))
        elif col.endswith("_max"):
            df[col] = df[col].fillna(df[col].max(skipna=True))
        else:
            # numeric general
            if np.issubdtype(df[col].dtype, np.number):
                df[col] = df[col].fillna(df[col].mean(skipna=True))
    return df
