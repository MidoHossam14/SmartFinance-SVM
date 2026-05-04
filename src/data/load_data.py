import os
import pandas as pd


def load_data(
    file_path: str = None,
    show_info: bool = False
) -> pd.DataFrame:
    """
    Load dataset from CSV file.

    Parameters:
    ----------
    file_path : str
        Path to the dataset. If None, default path is used.
    show_info : bool
        If True, prints dataset shape and info.

    Returns:
    -------
    df : pd.DataFrame
        Loaded dataset
    """

    # Default path
    if file_path is None:
        file_path = os.path.join(
            os.path.dirname(__file__),
            "../../data/raw/default_credit _card_clients.csv"
        )

    # Normalize path
    file_path = os.path.abspath(file_path)

    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Dataset not found at: {file_path}")

    # Load data
    df = pd.read_csv(file_path)

    if show_info:
        print("\n📊 Dataset Loaded Successfully!")
        print(f"Shape: {df.shape}")
        print("\nColumns:")
        print(df.columns.tolist())
        print("\nInfo:")
        print(df.info())

    return df


def load_processed_data(
    file_path: str = None
) -> pd.DataFrame:
    """
    Load processed dataset (after preprocessing).

    Parameters:
    ----------
    file_path : str
        Path to processed dataset

    Returns:
    -------
    df : pd.DataFrame
    """

    if file_path is None:
        file_path = os.path.join(
            os.path.dirname(__file__),
            "../../data/processed/processed_data.csv"
        )

    file_path = os.path.abspath(file_path)

    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Processed dataset not found at: {file_path}")

    return pd.read_csv(file_path)
