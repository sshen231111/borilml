from collections import Counter

import numpy as np
import pandas as pd


def load_csv_column(csv_file, column_index, first_row_header=False):
    df = pd.read_csv(csv_file, header=None if not first_row_header else 'infer')
    return df.iloc[:, column_index]


def load_csv_row(csv_file, row_index, first_row_header=False):
    df = pd.read_csv(csv_file, header=None if not first_row_header else 'infer')
    return df.iloc[row_index]


def load_data():
    reviews_matrix = pd.read_csv("../data/clean_review_data.csv").to_numpy().T
    review_dict = load_csv_row("../data/clean_unigram.csv", 0, first_row_header=False)
    sorted_indices = np.argsort(reviews_matrix[0])
    reviews_matrix = reviews_matrix[:, sorted_indices]
    return reviews_matrix, review_dict


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


def save_synthetic_text_to_csv(synthetic_text, file_name):
    # Count the occurrences of each word in the synthetic text
    word_counts = Counter(synthetic_text)

    # Convert the word counts to a DataFrame
    df = pd.DataFrame.from_dict(word_counts, orient='index', columns=['Count'])

    # Reset the index to make the words a column
    df.reset_index(inplace=True)
    df.columns = ['Word', 'Count']

    # Save the DataFrame to a CSV file
    df.to_csv(file_name, index=False)


def save_synthetic_text_to_txt(synthetic_text, file_name):
    # Count the occurrences of each word in the synthetic text
    word_counts = Counter(synthetic_text)

    # Create a list to hold the repeated words
    repeated_words = []

    # Repeat each word according to its count
    for word, count in word_counts.items():
        repeated_words.extend([word] * count)

    # Convert the list of repeated words to a comma-separated string
    text = ', '.join(repeated_words)

    # Save the string to a text file
    with open(file_name, 'w') as f:
        f.write(text)
