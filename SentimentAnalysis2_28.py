# External imports
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report, det_curve, DetCurveDisplay, recall_score
from sklearn.metrics.pairwise import cosine_similarity
from bs4 import BeautifulSoup
from numpy.linalg import norm
import csv
import string
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tabulate import tabulate

# Internal imports
from files import dictionary_maker, ngram_maker, csv_tasks, commands

DATA_PATH = "data/reviews/empty_removed_clean_review_data.csv"
UNI_DICTIONARY_PATH = "data/dictionaries/clean_unigram.csv"
BI_DICTIONARY_PATH = "data/dictionaries/unique_bigrams.csv"
PERCENTAGE_TESTING = 0.1
PERCENTAGE_TRAINING = 1 - PERCENTAGE_TESTING
INITIAL_PERCENT_FILTER = 0.2

# Define default parameters as a dictionary
DEFAULT_PARAMETERS = {
    "lower_threshold": -1,
    "upper_threshold": 1,
    "lower_bound": -0.1,
    "upper_bound": 0.1,
    "step_value": 0.02,
    "gram": "B",
}


def main():
    # Hyperparameters
    lower_threshold = DEFAULT_PARAMETERS["lower_threshold"]
    upper_threshold = DEFAULT_PARAMETERS["upper_threshold"]
    lower_bound = DEFAULT_PARAMETERS["lower_bound"]
    upper_bound = DEFAULT_PARAMETERS["upper_bound"]
    step_value = DEFAULT_PARAMETERS["step_value"]
    gram = DEFAULT_PARAMETERS["gram"]

    # Other variables
    pos_frequencies_train = None
    neg_frequencies_train = None
    pos_frequencies_test = None
    neg_frequencies_test = None
    EER = None
    thresholds = None
    confusion_matrices = None
    report = None

    # Load data
    if gram == "U":
        reviews_matrix, dictionary = csv_tasks.load_data(DATA_PATH, UNI_DICTIONARY_PATH)
    elif gram == "B":
        reviews_matrix, dictionary = csv_tasks.load_data(DATA_PATH, BI_DICTIONARY_PATH)
    else:
        print("Error with gram selection")
    print("System start up... Default hyperparameters set...")

    while True:
        # Store hyperparameters in a list of tuples
        hyperparameters = [
            ("Percentage for testing data", PERCENTAGE_TESTING),
            ("Percentage for training data", PERCENTAGE_TRAINING),
            ("Lower threshold value", lower_threshold),
            ("Upper threshold value", upper_threshold),
            ("Lower bound value", lower_bound),
            ("Upper bound value", upper_bound),
            ("Step value", step_value),
            ("Gram value", gram)
        ]

        print("Press 'H' for a list of commands.")
        command = input("Enter a command: ").strip().upper()

        if command == 'H':
            commands.print_commands()

        elif command == "F":
            report = commands.execute_filtered_main(reviews_matrix, dictionary, upper_threshold, lower_threshold,
                                                    PERCENTAGE_TESTING, gram)

        elif command == 'M':
            report = commands.execute_main(reviews_matrix, dictionary, upper_threshold, lower_threshold,
                                           PERCENTAGE_TESTING, gram)

        elif command == 'B':
            report = commands.execute_uni_bi_main(reviews_matrix, dictionary, upper_threshold, lower_threshold,
                                                  PERCENTAGE_TESTING, gram)


        elif command == 'A':
            report = commands.execute_filtered_squash_main(reviews_matrix, dictionary, upper_threshold, lower_threshold,
                                                           PERCENTAGE_TESTING, gram)

        elif command == 'I':
            # Update hyperparameters
            # lower_threshold, upper_threshold, lower_bound, upper_bound, step_value, commands.threshold_values = update_hyperparameters()
            print("Error in Parameter updating")

        elif command == 'P':
            # commands.print_data(hyperparameters, dictionary, pos_frequencies_train, neg_frequencies_train,
            #                     pos_frequencies_test, neg_frequencies_test, lower_threshold, upper_threshold,
            #                     EER, thresholds, confusion_matrices, report)
            print("must redo p")
        elif command == 'E':
            print("Program terminated.")
            break

        else:
            print("Invalid command. Please try again.")


main()
