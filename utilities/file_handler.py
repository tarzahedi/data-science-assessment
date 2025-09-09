import pandas as pd


def load_excel_file(data_path: str) -> pd.DataFrame:
    """Loads excel file as DataFrame

    Args:
        data_path (str): string path to the input file

    Raises:
        Exception: raises exception if multiple sheets exist in the file

    Returns:
        pd.DataFrame: DataFrame containing the excel data.
    """

    data = pd.ExcelFile(data_path)
    # check for number of sheets
    if len(data.sheet_names) == 1:
        return data.parse(data.sheet_names[0])
    else:
        raise Exception("Excel file should have exactly one sheet!")


def load_csv_file(data_path: str, sep: str | None = None) -> pd.DataFrame:
    """Load a CSV file into a pandas DataFrame.

    Args:
        data_path (str): Path to the CSV file to load.
        sep (str | None, optional): Delimiter to use. If None, pandas will
            infer the delimiter automatically. Defaults to None.

    Returns:
        pd.DataFrame: The contents of the CSV file as a pandas DataFrame.
    """
    if sep is not None:
        data = pd.read_csv(data_path, sep=sep)
    else:
        data = pd.read_csv(data_path)
    return data
