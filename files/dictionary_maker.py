import string

import pandas as pd
import numpy as np


def generate_review_dictionary(dictionary, reviews):
    """
    Generate a dictionary of words from reviews.

    Parameters:
    dictionary (list): List of words to consider.
    reviews (list): List of reviews.

    Returns:
    list: Sorted list of unique words from the reviews that are in the dictionary.
    """
    review_dict = []
    for review in reviews:
        words = review.split()
        for word in words:
            word_cleaned = word.lower().strip(string.punctuation)
            if word_cleaned in dictionary and word_cleaned not in review_dict:
                review_dict.append(word_cleaned)
    return sorted(review_dict)
