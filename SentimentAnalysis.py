import csv
import re
import string
from itertools import islice

import numpy as np
import pandas as pd
import scipy as sp
from scipy.sparse import csr_matrix
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report, det_curve, DetCurveDisplay, recall_score
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
from bs4 import BeautifulSoup
from numpy.linalg import norm
from tabulate import tabulate


########################################################################################################################
# CSV and file modification
########################################################################################################################

def load_csv_column(csv_file, column_index, first_row_header=False):
    """

    :param csv_file:
    :param column_index:
    :param first_row_header:
    :return:
    """
    df = pd.read_csv(csv_file, header=None if not first_row_header else 'infer')
    return df.iloc[:, column_index]


def load_csv_row(csv_file, row_index, first_row_header=False):
    """

    :param csv_file:
    :param row_index:
    :param first_row_header:
    :return:
    """
    df = pd.read_csv(csv_file, header=None if not first_row_header else 'infer')
    return df.iloc[row_index]


def load_data():
    """

    :return:
    """
    reviews_matrix = pd.read_csv("review_data.csv").to_numpy().T
    review_dict = load_csv_row("review_dict.csv", 0, first_row_header=False)
    review_bigram_dict = load_csv_row("review_bigram_dict.csv", 0, first_row_header=False)
    sorted_indices = np.argsort(reviews_matrix[0])
    reviews_matrix = reviews_matrix[:, sorted_indices]
    return reviews_matrix, review_dict, review_bigram_dict


def generate_bigrams(input_csv, output_csv):
    """

    :param input_csv:
    :param output_csv:
    """
    # Load the reviews from the CSV file
    reviews = pd.read_csv(input_csv, header=None)[1]

    # Generate bigrams for each review
    all_bigrams = []
    for review in reviews:
        if isinstance(review, str):  # Check if the review is a string
            words = review.split()  # Split the review into words
            cleaned_words = clean_words(words)  # Clean the words

            # Generate bigrams from cleaned words
            bigrams = generate_bigrams_from_words(cleaned_words)
            all_bigrams.extend(bigrams)

    # Write the bigrams to CSV
    write_data_to_csv(output_csv, all_bigrams)


def generate_bigrams_from_words(words):
    """

    :param words:
    :return:
    """
    bigrams = []
    for i in range(len(words) - 1):
        bigram = words[i] + " " + words[i + 1]
        bigrams.append(bigram)
    return bigrams


def write_data_to_csv(file_path, data):
    """
    Purpose: Write data to a CSV file with each bigram in its own column.

    Param(s):
        file_path (str): Path to the CSV file.
        data (list): Data to be written to the CSV file.

    Returns:
        None
    """
    with open(file_path, mode='w', newline='', encoding="utf-8") as csv_file:
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow(data)


########################################################################################################################
# Labeling
########################################################################################################################

def classify_labels(positive_scores, negative_scores, threshold):
    """

    :param positive_scores:
    :param negative_scores:
    :param threshold:
    :return:
    """
    labels = np.where((positive_scores - negative_scores - threshold) <= 0, 0, 1)
    scores = positive_scores - negative_scores
    return labels, scores


def label_classifier(pscore, nscore, threshold):
    """

    :param pscore:
    :param nscore:
    :param threshold:
    :return:
    """
    label = np.zeros(len(pscore), dtype=int)
    scores = np.zeros(len(pscore), dtype=float)
    # assigns labels as Positive(1) or Negative(0)
    for i in range(len(pscore)):
        prediction = pscore[i] - nscore[i] - threshold
        scores[i] = pscore[i] - nscore[i]
        if prediction <= 0:
            label[i] = 0
        else:
            label[i] = 1

    return label, scores


def create_actual_labels(ratings):
    """

    :param ratings:
    :return:
    """
    actual_labels = []
    for rating in ratings:
        if rating > 3:
            actual_labels.append(1)
        else:
            actual_labels.append(0)
    return actual_labels


########################################################################################################################
# Dictionary manipulation
########################################################################################################################
# def generate_review_dictionary(dictionary, reviews):
#     """
#
#     :param dictionary:
#     :param reviews:
#     :return:
#     """
#     review_dict = []
#     for review in reviews:
#         words = review.split()
#         for word in words:
#             word_cleaned = word.lower().strip(string.punctuation)
#             if word_cleaned in dictionary and word_cleaned not in review_dict:
#                 review_dict.append(word_cleaned)
#     return sorted(review_dict)


def clean_words(words):
    """
    Clean words by removing non-alphabetic characters and splitting based on special characters.

    :param words: List of words.
    :return: List of cleaned words.
    """
    cleaned_words = []
    for word in words:
        # Remove non-alphabetic characters
        word = ''.join(filter(str.isalpha, word))

        # Split the word based on special characters
        split_words = re.split(r'[.\-\/!@#$%^&*()_+={[}]|:;"<,>.?/~`]|(?<=[a-z])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])',
                               word)

        # If the word doesn't contain any special characters, simply append it
        if len(split_words) == 1 and split_words[0] != '':
            cleaned_words.append(split_words[0])
        else:
            cleaned_words.extend(split_words)

    return cleaned_words


def generate_bag_of_words_frequencies(dictionary, reviews, mode):
    """
    Generate bag of n-grams frequencies.

    :param dictionary: List of n-grams in the dictionary.
    :param reviews: List of review texts.
    :param mode: Mode for generating n-grams ("U" for unigrams, "B" for bigrams, "UB" for a mixture of both).
    :return: List of frequencies corresponding to the n-grams in the dictionary.
    """
    # Create a mapping from dictionary n-grams to their indices
    dictionary_mapping = {word: index for index, word in enumerate(dictionary)}

    # Initialize bag of n-grams
    bag_ngrams = np.zeros(len(dictionary))
    total_ngrams = 0

    # Iterate through each review
    for review_text in reviews:
        if isinstance(review_text, str) and review_text.lower() != 'nan':
            # Tokenize the review into n-grams based on the mode
            if mode == "U":
                ngrams = generate_ngrams(review_text, 1)
            elif mode == "B":
                ngrams = generate_ngrams(review_text, 2)
            elif mode == "UB":
                unigrams = generate_ngrams(review_text, 1)
                bigrams = generate_ngrams(review_text, 2)
                ngrams = unigrams + bigrams
            else:
                raise ValueError("Invalid mode. Mode must be 'U' for unigrams, 'B' for bigrams, or 'UB' for a mixture "
                                 "of both.")

            cleaned_ngrams = clean_words(ngrams)

            # Count the occurrence of each n-gram in the dictionary
            for ngram in cleaned_ngrams:
                index = dictionary_mapping.get(ngram)
                if index is not None:
                    total_ngrams += 1
                    bag_ngrams[index] += 1

    # Calculate frequencies
    return [(ngram_count / total_ngrams) for ngram_count in bag_ngrams] if total_ngrams > 0 else np.zeros(
        len(dictionary))


def generate_ngrams(text, n):
    """
    Generate n-grams from a given text.

    :param text: Input text.
    :param n: Size of n-grams.
    :return: List of n-grams.
    """
    words = text.split()
    ngrams = zip(*[islice(words, i, None) for i in range(n)])

    return [' '.join(ngram) for ngram in ngrams]


def generate_freq(review, dictionary, mode):
    """

    :param review:
    :param dictionary:
    :return:
    """
    neg_testing = review[review[:, 0] < 3]
    pos_testing = review[review[:, 0] > 3]
    pos_frequencies_test = None
    neg_frequencies_test = None

    pos_frequencies_test = generate_bag_of_words_frequencies(dictionary, pos_testing[:, 1], mode)
    neg_frequencies_test = generate_bag_of_words_frequencies(dictionary, neg_testing[:, 1], mode)

    pos_frequencies_test = np.array(pos_frequencies_test)
    pos_frequencies_test = pos_frequencies_test.reshape(-1, 1)

    neg_frequencies_test = np.array(neg_frequencies_test)
    neg_frequencies_test = neg_frequencies_test.reshape(-1, 1)

    return pos_frequencies_test, neg_frequencies_test


########################################################################################################################
# Matrix manipulation
########################################################################################################################
def rotate_2x2(matrix):
    """

    :param matrix:
    :return:
    """
    # Swap elements diagonally
    rotated_matrix = np.array([[matrix[1][1], matrix[1][0]],
                               [matrix[0][1], matrix[0][0]]])
    return rotated_matrix


########################################################################################################################
# Cosine similarity
########################################################################################################################

def cosine_similarity_scores(all_frequencies):
    """

    :param all_frequencies:
    :return:
    """
    positive_column = all_frequencies[:, 0].reshape(1, -1)
    negative_column = all_frequencies[:, 1].reshape(1, -1)
    pos_similarities = []
    neg_similarities = []
    for i in range(2, all_frequencies.shape[1]):
        test_column = all_frequencies[:, i].reshape(1, -1)
        positive_similarity = cosine_similarity(test_column, positive_column)[0, 0]
        negative_similarity = cosine_similarity(test_column, negative_column)[0, 0]
        pos_similarities.append(positive_similarity)
        neg_similarities.append(negative_similarity)
    return pos_similarities, neg_similarities


########################################################################################################################
# Performance evaluation
########################################################################################################################

# Calculations
def calculate_performance_matrix(inputdata, flag):
    """

    :param inputdata:
    :param flag:
    :return:
    """
    class_accuracy_matrix = np.empty(len(inputdata))
    for i in range(len(inputdata)):
        class_accuracy = (inputdata[i, i] / (np.sum(inputdata[i, :]))) * 100
        class_accuracy_matrix[i] = + class_accuracy

    Uar = round((1 / len(class_accuracy_matrix)) * (np.sum(class_accuracy_matrix)), 2)
    if flag == 1:
        print("Unweighted Accuracy:", Uar)

    # Proportion of true positive predictions in all positive predictions
    if np.sum(inputdata[:, 0]) == 0:
        Precision = 0
    else:
        Precision = round((inputdata[0, 0] / np.sum(inputdata[:, 0])), 2)
        if flag == 1:
            print("Precision:", Precision)
    # Proportion of true positive predictions made by the model out of all positive samples in the dataset
    if np.sum(inputdata[0, :]) == 0:
        Recall = 0
    else:
        Recall = round((inputdata[0, 0] / np.sum(inputdata[0, :])), 2)
        if flag == 1:
            print("Recall:", Recall)
    # Defined as the harmonic mean of precision and recall
    F1_Score = round((2 / ((1 / Precision) + (1 / Recall))), 2) if Precision != 0 and Recall != 0 else 0
    if flag == 1:
        print("F1 Score:", F1_Score, "\n")

    return Uar


def count_correct_predictions(actual_labels, predicted_labels):
    """

    :param actual_labels:
    :param predicted_labels:
    :return:
    """
    correct_count = 0
    wrong_count = 0
    for actual, predicted in zip(actual_labels, predicted_labels):
        if actual == predicted:
            correct_count += 1
        else:
            wrong_count += 1
    return correct_count, wrong_count


def percent_filter_calc(actual_labels, pos_frequencies_test, neg_frequencies_test, percent_filter_array, mask,
                        all_frequencies_test, upper_threshold, lower_threshold):
    """

    :param actual_labels:
    :param pos_frequencies_test:
    :param neg_frequencies_test:
    :param percent_filter_array:
    :param mask:
    :param all_frequencies_test:
    :param upper_threshold:
    :param lower_threshold:
    :return:
    """
    confusion_matrices = []
    uar_matrix = []
    all_fpr = []
    all_fnr = []
    all_EER = []
    all_pos = []
    all_neg = []
    for i, percent_filtered in enumerate(percent_filter_array):
        filtered_mask = calculate_filter(mask, percent_filtered)
        # filtered_mask = calculate_filter_ends(mask, percent_filtered)
        masked_neg = filtered_mask * neg_frequencies_test
        masked_pos = filtered_mask * pos_frequencies_test
        nonzero_neg = non_zero_values(masked_neg)
        nonzero_pos = non_zero_values(masked_pos)

        combined_matrix = np.concatenate((masked_pos, masked_neg, all_frequencies_test), axis=1)
        pos_cosine, neg_cosine = cosine_similarity_scores(combined_matrix)

        z, z, best_threshold = calculate_threshold_bisectional(pos_cosine,
                                                               neg_cosine,
                                                               upper_threshold,
                                                               lower_threshold,
                                                               actual_labels)
        # calculate predicted label and scores
        predicted_labels, calc_scores = label_classifier(pos_cosine, neg_cosine, best_threshold)
        # calculate the Confusion Matrix
        cm = metrics.confusion_matrix(actual_labels, predicted_labels)
        cm = rotate_2x2(cm)
        Uar = calculate_performance_matrix(cm, 0)
        fpr, fnr = calculate_fpr_fnr(cm)
        EER = (fpr + fnr) / 2

        confusion_matrices.append(cm)  # Append the confusion matrix
        uar_matrix.append(Uar)
        all_fpr.append(fpr)  # Accumulate false positive rates
        all_fnr.append(fnr)  # Accumulate false negative rates
        all_EER.append(EER)
        all_neg.append(nonzero_neg)
        all_pos.append(nonzero_pos)
        print("End of Iteration: ", i, "The percent tested: ", percent_filtered)
        print("The EER: ", EER)
        print("Non-Zero Size Negative: ", nonzero_neg)
        print("Non-Zero Size Positive: ", nonzero_pos)

    return confusion_matrices, uar_matrix, all_fpr, all_fnr, all_EER, all_neg, all_pos


def calculate_metrics(actual_labels, positive_scores, negative_scores, threshold):
    """

    :param actual_labels:
    :param positive_scores:
    :param negative_scores:
    :param threshold:
    :return:
    """
    # calculate predicted label and scores
    predicted_labels, calc_scores = label_classifier(positive_scores, negative_scores, threshold)
    # calculate the Confusion Matrix
    cm = metrics.confusion_matrix(actual_labels, predicted_labels)
    cm = rotate_2x2(cm)
    Uar = calculate_performance_matrix(cm, 0)
    fpr, fnr = calculate_fpr_fnr(cm)
    EER = (fpr + fnr) / 2

    return predicted_labels, calc_scores, cm, EER, fpr, fnr, Uar


def calculate_EER(fpr, fnr):
    """

    :param fpr:
    :param fnr:
    :return:
    """
    EER_Matrix = np.empty(len(fpr))
    for i in range(len(fpr)):
        eer = (fpr[i] + fnr[i]) / 2.0
        EER_Matrix[i] = eer
    best_eer = min(EER_Matrix)
    return best_eer


def find_max(list):
    """

    :param list:
    :return:
    """
    max_value = max(list)
    max_index = list.index(max_value)

    return max_index


def statistics(actual_labels, predicted_labels):
    """

    :param actual_labels:
    :param predicted_labels:
    :return:
    """
    # Generate a classification report
    report = classification_report(actual_labels, predicted_labels, zero_division=1)
    return report


# Calculates the weight mask and returns it
def calculate_mask(negative_freq, positive_freq):
    """
    Calculate mask for filtering based on the difference between positive and negative frequencies.

    :param negative_freq: Frequencies of n-grams in negative reviews (sparse matrix).
    :param positive_freq: Frequencies of n-grams in positive reviews (sparse matrix).
    :return: Mask array for filtering.
    """
    # Compute delta and summation
    delta = abs(positive_freq - negative_freq)
    summation = negative_freq + positive_freq

    # Replace 0 elements with 1 to avoid division by zero
    modified_array = np.where(summation == 0, 1, summation)

    # Calculate weights
    weights = delta / modified_array
    return weights


def calculate_filter(mask, percent_filter):
    """

    :param mask:
    :param percent_filter:
    :return:
    """
    sorted_mask = np.sort(mask)
    percent_filter = np.percentile(sorted_mask, (percent_filter / 2) * 100)
    inverse_filter = np.percentile(sorted_mask, (100 - ((percent_filter / 2) * 100)))

    mask[mask < percent_filter] = 0
    mask[mask > inverse_filter] = 0
    return mask


def calculate_filter_ends(mask, percent_filter):
    """

    :param mask:
    :param percent_filter:
    :return:
    """
    upper_percent = (percent_filter * 3) / 4
    lower_percent = percent_filter / 4

    upper_percent = np.percentile(mask, (upper_percent) * 100)
    lower_percent = np.percentile(mask, (lower_percent) * 100)

    mask[mask < lower_percent] = 0
    mask[mask > upper_percent] = 0
    return mask


def calculate_threshold_bisectional(pos_cosine, neg_cosine, upper_threshold, lower_threshold, actual_labels):
    """

    :param pos_cosine:
    :param neg_cosine:
    :param upper_threshold:
    :param lower_threshold:
    :param actual_labels:
    :return:
    """
    count = 0
    mid_threshold = (upper_threshold + lower_threshold) / 2
    mid_predicted_label, mid_calc_score = label_classifier(pos_cosine, neg_cosine, mid_threshold)

    cm_mid = metrics.confusion_matrix(actual_labels, mid_predicted_label)
    fpr_mid, fnr_mid = calculate_fpr_fnr(cm_mid)
    diff_mid = fpr_mid - fnr_mid

    if diff_mid > 0:
        lower_threshold = mid_threshold
    elif diff_mid < 0:
        upper_threshold = mid_threshold

    while (count < 1000):
        count += 1
        mid_threshold = (upper_threshold + lower_threshold) / 2
        mid_predicted_label, mid_calc_score = label_classifier(pos_cosine, neg_cosine, mid_threshold)

        cm_mid = metrics.confusion_matrix(actual_labels, mid_predicted_label)
        fpr_mid, fnr_mid = calculate_fpr_fnr(cm_mid)
        # print("This current FPR and FNR: ", fpr_mid, fnr_mid)
        diff_mid = fpr_mid - fnr_mid

        if diff_mid > 0:
            lower_threshold = mid_threshold
        elif diff_mid < 0:
            upper_threshold = mid_threshold
        # print("The lower and upper thresholds: ",lower_threshold, upper_threshold)

        if abs(lower_threshold - upper_threshold) <= 0.0000002:
            break

    mid_threshold = (upper_threshold + lower_threshold) / 2
    return upper_threshold, lower_threshold, mid_threshold


def binary_search(low, mid, high):
    """

    :param low:
    :param mid:
    :param high:
    :return:
    """
    if abs(low - mid) < abs(high - mid):
        return low, mid
    else:
        return mid, high


def calculate_fpr_fnr(confusion_matrix):
    """

    :param confusion_matrix:
    :return:
    """
    # Extract values from confusion matrix
    TN, FP, FN, TP = confusion_matrix.ravel()
    # Calculate False Positive Rate (FPR)
    if FP + TN != 0:
        FPR = FP / (FP + TN)
    else:
        FPR = 0
    # Calculate False Negative Rate (FNR)
    if FN + TP != 0:
        FNR = FN / (FN + TP)
    else:
        FNR = 0
    return FPR, FNR


def non_zero_values(frequencies):
    """

    :param frequencies:
    :return:
    """
    num_non_zero = np.count_nonzero(frequencies)
    return num_non_zero


########################################################################################################################
# Graphing
########################################################################################################################

def calculate_det(fpr, fnr):
    """

    :param fpr:
    :param fnr:
    """
    display = DetCurveDisplay(fpr=fpr, fnr=fnr, estimator_name="DET Curve")
    display.plot()
    plt.xlabel('False Positive Rate (FPR)')
    plt.ylabel('False Negative Rate (FNR)')
    plt.title('Detection Error Tradeoff (DET) Curve')
    plt.show()


def plot_metrics(percent_filter_array, all_fpr, all_fnr, all_EER):
    """

    :param percent_filter_array:
    :param all_fpr:
    :param all_fnr:
    :param all_EER:
    """
    plt.figure(figsize=(10, 6))
    plt.plot(percent_filter_array, all_fpr, label='FPR')
    plt.plot(percent_filter_array, all_fnr, label='FNR')
    plt.plot(percent_filter_array, all_EER, label='EER')

    plt.title('Metrics vs. Percent Filter')
    plt.xlabel('Percent Filter')
    plt.ylabel('EER Value')
    plt.legend()
    plt.grid(True)
    plt.show()


def plot_nonzero(percent_filter_array, all_neg, all_pos):
    """

    :param percent_filter_array:
    :param all_neg:
    :param all_pos:
    """
    fig, axes = plt.subplots(2, 1, figsize=(10, 10))

    # Plot for positive array
    axes[0].plot(percent_filter_array, all_pos, label='positive', color='blue')
    axes[0].set_title('Positive Array')
    axes[0].set_xlabel('Percent Filter')
    axes[0].set_ylabel('Frequencies in Array Value')
    axes[0].grid(True)
    axes[0].legend()

    # Plot for negative array
    axes[1].plot(percent_filter_array, all_neg, label='negative', color='orange')
    axes[1].set_title('Negative Array')
    axes[1].set_xlabel('Percent Filter')
    axes[1].set_ylabel('Frequencies in Array Value')
    axes[1].grid(True)
    axes[1].legend()

    plt.tight_layout()
    plt.show()


def plot_threshold_vs_accuracy(threshold_values, uar_values):
    """

    :param threshold_values:
    :param uar_values:
    """
    # Plotting the threshold_matrix(x axis) vs unweighted accuracy (UAR)
    plt.figure(figsize=(8, 5))
    plt.plot(threshold_values, uar_values, marker='o', linestyle='-')
    plt.title('Threshold vs Unweighted Accuracy')
    plt.xlabel('Threshold')
    plt.ylabel('Unweighted Accuracy (UAR)')
    plt.grid(True)
    plt.show()


########################################################################################################################
# Console logging
########################################################################################################################
def print_best_threshold(cm, EER, FPR, FNR, UAR, threshold):
    """

    :param cm:
    :param EER:
    :param FPR:
    :param FNR:
    :param UAR:
    :param threshold:
    """
    print("The Threshold Used:", threshold)
    print(f"Best Possible Confusion Matrix:\n{cm}")
    print("Best Possible FPR: ", FPR)
    print("Best Possible FNR: ", FNR)
    print("Best Possible Equal Error Rate: ", EER)
    print("Best Possible Unweighted Accuracy: ", UAR)


########################################################################################################################
# Main
########################################################################################################################
def main():
    """

    """
    # Hyperparameters
    percentage_testing = 0.1
    percentage_training = 1 - percentage_testing
    lower_threshold = None
    upper_threshold = None
    lower_bound = None
    upper_bound = None
    step_value = None
    mode = None

    # Other variables
    pos_frequencies_train = None
    neg_frequencies_train = None
    pos_frequencies_test = None
    neg_frequencies_test = None
    actual_labels = None
    predicted_labels = None
    EER = None
    threshold_values = None
    thresholds = None
    confusion_matrices = None
    report = None
    uar_matrix = None
    all_fpr = None
    all_fnr = None
    parameters_set = False

    # Load data
    reviews_matrix, review_dict, review_bigram_dict = load_data()

    while True:
        if not parameters_set:
            print("Setting hyperparameters...")
            command = input("Enter Y for default settings or any other key for manual: ").strip().upper()
            if command == "Y":
                lower_threshold = -1
                upper_threshold = 1
                lower_bound = -0.1
                upper_bound = 0.1
                step_value = 0.02
                mode = "U"
            else:
                lower_threshold = float(input("Enter lower threshold value (Recommended value=-1): "))
                upper_threshold = float(input("Enter upper threshold value (Recommended value=1): "))
                lower_bound = float(input("Enter lower bound value (Recommended value=-0.1, an integer or float that's "
                                          "less than zero): "))
                upper_bound = float(input("Enter upper bound value (Recommended value=0.1, an integer or float that's "
                                          "greater than zero): "))
                step_value = float(input("Enter step value (Recommended value=0.02): "))
                threshold_values = np.arange(lower_bound, upper_bound, step_value)
                mode = str(input("Enter a mode for the system (U = Unigrams, B = Bigrams, UB = Mixture of Both)"))

                while mode.strip().upper() != "U" and mode.strip().upper() != "B" and mode.strip().upper() != "UB":
                    mode = str(input("Invalid input! Please enter a mode for the system (U = Unigrams, B = Bigrams, "
                                     "UB = Mixture of Both): "))

            # Store hyperparameters in a list of tuples
            hyperparameters = [
                ("Percentage for testing data", percentage_testing),
                ("Percentage for training data", percentage_training),
                ("Lower threshold value", lower_threshold),
                ("Upper threshold value", upper_threshold),
                ("Lower bound value", lower_bound),
                ("Upper bound value", upper_bound),
                ("Step value", step_value),
                ("Mode", mode)
            ]

            parameters_set = True

        print("Press 'H' for a list of commands.")
        command = input("Enter a command: ").strip().upper()

        if command == 'H':
            print("Commands:")
            print("R - Run main function of the program")
            print("G - Graph generation")
            print("I - Input new hyperparameters")
            print("P - Print values")
            print("E - Terminate program")

        elif command == 'R':
            print("Executing main function...")

            if (percentage_testing is not None and lower_threshold is not None and upper_threshold is not None
                    and lower_bound is not None and upper_bound is not None and step_value is not None):
                sorted_indices = np.argsort(reviews_matrix[0])
                reviews_matrix = reviews_matrix[:, sorted_indices]
                review_training, review_testing = train_test_split(reviews_matrix.T, test_size=percentage_testing,
                                                                   random_state=42)

                neg_testing = review_testing[review_testing[:, 0] < 3]
                pos_testing = review_testing[review_testing[:, 0] > 3]

                dictionary = None

                if mode == "U":
                    dictionary = review_dict
                elif mode == "B":
                    dictionary = review_bigram_dict
                elif mode == "UB":
                    dictionary = pd.concat([review_bigram_dict, review_dict], ignore_index=True)

                pos_frequencies_test, neg_frequencies_test = generate_freq(review_training, dictionary, mode)

                # print(pos_testing.shape)
                # print(neg_testing.shape)

                combined_testing = np.concatenate((pos_testing, neg_testing), axis=0)

                all_frequencies_test = []

                # print(len(combined_testing))
                # print(dictionary.shape)

                run = 0

                for i in range(len(combined_testing)):
                    run += 1
                    review_rate = combined_testing[i][0]
                    review_text = combined_testing[i][1]
                    if review_text == "":
                        continue

                    try:
                        print("Run succeeded: " + str(run))
                        review_freq = generate_bag_of_words_frequencies(dictionary, [review_text], mode)
                        # Append review_freq to the list
                        all_frequencies_test.append(review_freq)
                    except Exception as e:
                        print("Run failed:", run)
                        print("Reason for failure:", str(e))
                        break

                all_frequencies_test = np.array(all_frequencies_test).T
                actual_labels = create_actual_labels(combined_testing[:, 0])

                print(len(actual_labels))

                # if .46 is entered then Any values in the mask array below the 23rd percentile would be set to 0.
                # Any values above the 77th percentile would also be set to 0. In essence, this function would retain
                # only the values within the middle 54% of the data range, setting the lowest 23% and the highest 23%
                # of values to 0.
                percent_filter = 0.46
                percent_filter_array = np.arange(0.4, 0.8, 0.01)
                mask = calculate_mask(neg_frequencies_test, pos_frequencies_test)
                print("The non zero values in mask:", non_zero_values(neg_frequencies_test))
                print("The non zero values in mask:", non_zero_values(pos_frequencies_test))
                # 11,774 Total
                # Start with 8559 nonzero terms- 73% are nonzero terms
                # EER Diverges around 6501 nonzero terms- 55% are nonzero terms
                filtered_mask = calculate_filter(mask, percent_filter)
                masked_neg = filtered_mask * neg_frequencies_test
                masked_pos = filtered_mask * pos_frequencies_test

                print(masked_pos.shape)
                print(masked_neg.shape)
                print(all_frequencies_test.shape)

                combined_matrix = np.concatenate((masked_pos, masked_neg, all_frequencies_test), axis=1)

                # combined_matrix = np.concatenate((pos_frequencies_test, neg_frequencies_test, all_frequencies_test), axis=1)
                pos_cosine, neg_cosine = cosine_similarity_scores(combined_matrix)

                print("///////////////////////////////////////////////////////////////////////")
                print(len(pos_cosine))
                print(len(neg_cosine))
                print(upper_threshold)
                print(lower_threshold)
                print(len(actual_labels))
                print("///////////////////////////////////////////////////////////////////////")

                upper_threshold, lower_threshold, best_threshold = calculate_threshold_bisectional(pos_cosine,
                                                                                                   neg_cosine,
                                                                                                   upper_threshold,
                                                                                                   lower_threshold,
                                                                                                   actual_labels)
                confusion_matrices, uar_matrix, all_fpr, all_fnr, all_EER, all_neg, all_pos = (
                    percent_filter_calc(actual_labels, pos_frequencies_test, neg_frequencies_test, percent_filter_array,
                                        mask, all_frequencies_test, 100, -100))

                # threshold_matrix = np.arange(best_threshold - 0.5, best_threshold + 0.5, 0.002).reshape(-1, 1)

                predicted_labels, calc_scores, cm, EER, fpr, fnr, Uar = calculate_metrics(actual_labels, pos_cosine,
                                                                                          neg_cosine, best_threshold)
                print_best_threshold(cm, EER, fpr, fnr, Uar, best_threshold)

                plot_metrics(percent_filter_array, all_fpr, all_fnr, all_EER)
                plot_nonzero(percent_filter_array, all_neg, all_pos)

                report = statistics(actual_labels, predicted_labels)
                print("The non zero values in mask:", non_zero_values(mask))

            else:
                print("Hyperparameters are not set. Please set hyperparameters first.")

        elif command == 'G':
            print("Generating graphs...")

            if thresholds is not None and uar_matrix is not None and all_fpr is not None and all_fnr is not None:
                plot_threshold_vs_accuracy(thresholds, uar_matrix)
                calculate_det(all_fpr, all_fnr)
            else:
                print("Graphs cannot be generated. Required data is missing.")

        elif command == 'I':
            print("Setting hyperparameters...")

            lower_threshold = float(input("Enter lower threshold value (Recommended value=-1): "))
            upper_threshold = float(input("Enter upper threshold value (Recommended value=1): "))
            lower_bound = float(input("Enter lower bound value (Recommended value=-0.1, an integer or float that's "
                                      "less than zero): "))
            upper_bound = float(input("Enter upper bound value (Recommended value=0.1, an integer or float that's "
                                      "greater than zero): "))
            step_value = float(input("Enter step value (Recommended value=0.02): "))
            threshold_values = np.arange(lower_bound, upper_bound, step_value)
            mode = str(input("Enter a mode for the system (U = Unigrams, B = Bigrams, UB = Mixture of Both)"))

            while mode.strip().upper() != "U" and mode.strip().upper() != "B" and mode.strip().upper() != "UB":
                mode = str(input("Invalid input! Please enter a mode for the system (U = Unigrams, B = Bigrams, "
                                 "UB = Mixture of Both): "))

            print("Hyperparameters set successfully...")

        elif command == 'P':
            print("Printing data...\n")

            # Print the hyperparameters
            print("Current parameters:")
            if hyperparameters:
                print(tabulate(hyperparameters, headers=["Parameter", "Value"], tablefmt="grid"))
            else:
                print("Hyperparameters are not set.")

            # Check the length of the dictionary
            print("\nLength of the dictionary:", len(review_dict))

            # Print the bag-of-words frequencies if they are set
            if pos_frequencies_train is not None:
                print("\nBag-of-words frequencies (Positive):")
                print(pos_frequencies_train[500:800])
            if neg_frequencies_train is not None:
                print("\nBag-of-words frequencies (Negative):")
                print(neg_frequencies_train[800:1000])

            # Print the frequencies for testing data if they are set
            if pos_frequencies_test is not None:
                print("\nPositive Reviews Frequencies (first 5):")
                print(pos_frequencies_test[:5])
            if neg_frequencies_test is not None:
                print("\nNegative Reviews Frequencies (first 5):")
                print(neg_frequencies_test[:5])

            # Print threshold values and EER if they are set
            if lower_threshold is not None and upper_threshold is not None:
                print("\nThe bounds are:", lower_threshold, "to", upper_threshold)
            if EER is not None:
                print("Equal Error Rate:", EER)

            # Print peak results if available
            if thresholds is not None and confusion_matrices is not None:
                if len(thresholds) > 1:
                    max_index = find_max(uar_matrix)
                    print("\nThe peak results are:\n")
                    print("Threshold:", thresholds[max_index])
                    print("Confusion Matrix:")
                    headers = ["", "Expected Positive", "Expected Negative"]
                    matrix_data = [["", "P", "N"],
                                   ["P", confusion_matrices[max_index][1, 1], confusion_matrices[max_index][1, 0]],
                                   ["N", confusion_matrices[max_index][0, 1], confusion_matrices[max_index][0, 0]]]
                    print(tabulate(matrix_data, headers=headers, tablefmt="grid"))
                    calculate_performance_matrix(confusion_matrices[max_index], 1)
                else:
                    print("\nThe peak results are:\n")
                    print("Threshold:", thresholds[0])
                    print("Confusion Matrix:")
                    headers = ["", "Expected Positive", "Expected Negative"]
                    matrix_data = [["", "P", "N"],
                                   ["P", confusion_matrices[0][1, 1], confusion_matrices[0][1, 0]],
                                   ["N", confusion_matrices[0][0, 1], confusion_matrices[0][0, 0]]]
                    print(tabulate(matrix_data, headers=headers, tablefmt="grid"))
                    calculate_performance_matrix(confusion_matrices[0], 1)
            else:
                print("\nPeak results are not available.")

            # Print the report if it is available
            if report is not None:
                print("\nClassification Report:")
                print(report)

        elif command == 'E':
            print("Program terminated.")
            break

        else:
            print("Invalid command. Please try again.")


main()
