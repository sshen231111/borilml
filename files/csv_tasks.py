import pandas as pd
import numpy as np


def save_to_csv(array, file_name):
    """
    Save the given array to a CSV file.

    Parameters:
        array (list or numpy array): The array to be saved.
        file_name (str): The name of the CSV file (include .csv extension).

    Returns:
        None
    """
    # Convert the array to a pandas DataFrame
    df = pd.DataFrame(array)

    # Write the DataFrame to the CSV file
    df.to_csv(file_name, index=False)


def load_csv_column(csv_file, column_index, first_row_header=False):
    """
    Load a specific column from a CSV file.

    Parameters:
    csv_file (str): Path to the CSV file.
    column_index (int): Index of the column to load.
    first_row_header (bool): Whether the first row of the CSV file is a header.

    Returns:
    pandas.Series: The specified column from the CSV file.
    """
    df = pd.read_csv(csv_file, header=None if not first_row_header else 'infer')
    return df.iloc[:, column_index]


def load_csv_row(csv_file, row_index, first_row_header=False):
    """
    Load a specific row from a CSV file.

    Parameters:
    csv_file (str): Path to the CSV file.
    row_index (int): Index of the row to load.
    first_row_header (bool): Whether the first row of the CSV file is a header.

    Returns:
    pandas.Series: The specified row from the CSV file.
    """
    df = pd.read_csv(csv_file, header=None if not first_row_header else 'infer')
    return df.iloc[row_index]


def load_data(data_path, dictionary_path):
    """
    Load review matrix and dictionary from specified paths.

    Parameters:
    data_path (str): Path to the data file.
    dictionary_path (str): Path to the dictionary file.

    Returns:
    tuple: The review matrix and dictionary.
    """
    reviews_matrix = pd.read_csv(data_path).to_numpy().T
    review_dict = load_csv_row(dictionary_path, 0, first_row_header=False)
    sorted_indices = np.argsort(reviews_matrix[0])
    reviews_matrix = reviews_matrix[:, sorted_indices]
    return reviews_matrix, review_dict