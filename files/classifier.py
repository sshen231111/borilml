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


def generate_freq(review, review_dict):
    """
    Generate frequency of positive and negative reviews.

    Parameters:
    review (numpy.ndarray): The review data.
    review_dict (dict): The review dictionary.

    Returns:
    tuple: A tuple containing frequencies of positive and negative reviews.
    """
    neg_testing = review[review[:, 0] < 3]
    pos_testing = review[review[:, 0] > 3]

    pos_frequencies_test = generate_bag_of_words_frequencies(review_dict, pos_testing[:, 1])
    pos_frequencies_test = np.array(pos_frequencies_test)
    pos_frequencies_test = pos_frequencies_test.reshape(-1, 1)

    neg_frequencies_test = generate_bag_of_words_frequencies(review_dict, neg_testing[:, 1])
    neg_frequencies_test = np.array(neg_frequencies_test)
    neg_frequencies_test = neg_frequencies_test.reshape(-1, 1)
    return pos_frequencies_test, neg_frequencies_test


def generate_bag_of_words_frequencies(dictionary, reviews):
    """
    Generate bag of words frequencies for given reviews.

    Parameters:
    dictionary (dict): The dictionary of words.
    reviews (list): The list of reviews.

    Returns:
    list: A list of word frequencies.
    """
    dictionary_mapping = {word: index for index, word in enumerate(dictionary)}
    bag_words = np.zeros(len(dictionary))
    total_words = 0
    for review_text in reviews:
        if isinstance(review_text, str) and review_text.lower() != 'nan':
            words = review_text.split()  # Split review text into words
            cleaned_words = [word.lower().strip(string.punctuation) for word in words]
            for word in cleaned_words:
                index = dictionary_mapping.get(word)
                if index is not None:
                    total_words += 1
                    bag_words[index] += 1
    return [(word_count / total_words) for word_count in bag_words] if total_words > 0 else np.zeros(len(dictionary))


def cosine_similarity_scores(all_frequencies):
    """
    Calculate cosine similarity scores for all frequencies.

    Parameters:
    all_frequencies (numpy.ndarray): The frequency data.

    Returns:
    tuple: A tuple containing lists of positive and negative similarities.
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


def classify_labels(positive_scores, negative_scores, threshold):
    """
    Classify labels based on positive scores, negative scores, and a threshold.

    Parameters:
    positive_scores (list): The list of positive scores.
    negative_scores (list): The list of negative scores.
    threshold (float): The threshold value for classification.

    Returns:
    tuple: A tuple containing labels and scores.
    """
    labels = np.where((positive_scores - negative_scores - threshold) <= 0, 0, 1)
    scores = positive_scores - negative_scores
    return labels, scores


def label_classifier(pscore, nscore, threshold):
    """
    Classify labels based on positive scores, negative scores, and a threshold.

    Parameters:
    pscore (list): The list of positive scores.
    nscore (list): The list of negative scores.
    threshold (float): The threshold value for classification.

    Returns:
    tuple: A tuple containing labels and scores.
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


def calculate_performance_matrix(inputdata, flag):
    """
    Calculate performance matrix for given input data.

    Parameters:
    inputdata (numpy.ndarray): The input data.
    flag (int): The flag to indicate whether to print the results.

    Returns:
    float: The unweighted accuracy.
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


def calculate_det(fpr, fnr):
    """
    Calculate and plot the Detection Error Tradeoff (DET) curve.

    Parameters:
    fpr (list): The list of false positive rates.
    fnr (list): The list of false negative rates.
    """
    display = DetCurveDisplay(fpr=fpr, fnr=fnr, estimator_name="DET Curve")
    display.plot()
    plt.xlabel('False Positive Rate (FPR)')
    plt.ylabel('False Negative Rate (FNR)')
    plt.title('Detection Error Tradeoff (DET) Curve')
    plt.show()


def count_correct_predictions(actual_labels, predicted_labels):
    """
    Count the number of correct and wrong predictions.

    Parameters:
    actual_labels (list): The list of actual labels.
    predicted_labels (list): The list of predicted labels.

    Returns:
    tuple: A tuple containing counts of correct and wrong predictions.
    """
    correct_count = 0
    wrong_count = 0
    for actual, predicted in zip(actual_labels, predicted_labels):
        if actual == predicted:
            correct_count += 1
        else:
            wrong_count += 1
    return correct_count, wrong_count


def confusion_matrix_scheduler(actual_labels, positive_scores, negative_scores, thresholds):
    """
    Schedule the calculation of confusion matrix for given labels, scores, and thresholds.

    Parameters:
    actual_labels (list): The list of actual labels.
    positive_scores (list): The list of positive scores.
    negative_scores (list): The list of negative scores.
    thresholds (list): The list of thresholds.

    Returns:
    tuple: A tuple containing predicted labels, equal error rate, thresholds, confusion matrices, unweighted accuracy matrix, false positive rates, and false negative rates.
    """
    uar_matrix = []
    confusion_matrices = []  # Initialize an empty list to store confusion matrices

    all_fpr = []
    all_fnr = []

    for i, threshold in enumerate(thresholds):
        # calculate predicted label and scores
        predicted_labels, calc_scores = label_classifier(positive_scores, negative_scores, threshold)
        # calculate the Confusion Matrix
        cm = metrics.confusion_matrix(actual_labels, predicted_labels)
        cm = rotate_2x2(cm)
        Uar = calculate_performance_matrix(cm, 0)
        fpr, fnr, _ = det_curve(actual_labels, calc_scores)

        confusion_matrices.append(cm)  # Append the confusion matrix
        uar_matrix.append(Uar)
        all_fpr.extend(fpr)  # Accumulate false positive rates
        all_fnr.extend(fnr)  # Accumulate false negative rates

    EER = calculate_EER(all_fpr, all_fnr)
    return predicted_labels, EER, thresholds, confusion_matrices, uar_matrix, all_fpr, all_fnr


def find_max(list_of_values):
    """
    Find the maximum value in a list and its index.

    Parameters:
    list_of_values (list): The list of values.

    Returns:
    int: The index of the maximum value.
    """
    max_value = max(list_of_values)
    max_index = list_of_values.index(max_value)

    return max_index


def plot_threshold_vs_accuracy(threshold_values, uar_values):
    """
    Plot threshold values versus unweighted accuracy.

    Parameters:
    threshold_values (list): The list of threshold values.
    uar_values (list): The list of unweighted accuracy values.
    """
    # Plotting the threshold_matrix(x axis) vs unweighted accuracy (UAR)
    plt.figure(figsize=(8, 5))
    plt.plot(threshold_values, uar_values, marker='o', linestyle='-')
    plt.title('Threshold vs Unweighted Accuracy')
    plt.xlabel('Threshold')
    plt.ylabel('Unweighted Accuracy (UAR)')
    plt.grid(True)
    plt.show()


def statistics(actual_labels, predicted_labels):
    """
    Generate a classification report for actual and predicted labels.

    Parameters:
    actual_labels (list): The list of actual labels.
    predicted_labels (list): The list of predicted labels.

    Returns:
    str: The classification report.
    """
    # Generate a classification report
    report = classification_report(actual_labels, predicted_labels, zero_division=1)
    return report


# Calculates the weight mask and returns it
def calculate_mask(negative_freq, positive_freq):
    """
    Calculate the weight mask for negative and positive frequencies.

    Parameters:
    negative_freq (numpy.ndarray): The negative frequencies.
    positive_freq (numpy.ndarray): The positive frequencies.

    Returns:
    numpy.ndarray: The weight mask.
    """
    delta = abs(positive_freq - negative_freq)
    summation = negative_freq + positive_freq
    modified_array = np.where(summation == 0, 1, summation)
    weights = delta / modified_array
    return weights


def calculate_filter(mask_temp, percent_filter):
    """
    Calculate the filter for a mask.

    Parameters:
    mask_temp (numpy.ndarray): The mask.
    percent_filter (float): The percentage of elements to filter.

    Returns:
    tuple: A tuple containing the filtered mask and the number of elements set to zero.
    """
    num_elements_to_zero = int(len(mask_temp) * percent_filter)
    # Sort the array and find the threshold value
    sorted_values = np.sort(mask_temp, axis=0)
    filter_threshold = sorted_values[num_elements_to_zero]
    # Set elements less than the threshold to zero
    mask_temp[mask_temp < filter_threshold] = 0
    return mask_temp, num_elements_to_zero


def calculate_EER(fpr, fnr):
    """
    Calculate the Equal Error Rate (EER).

    Parameters:
    fpr (list): The list of false positive rates.
    fnr (list): The list of false negative rates.

    Returns:
    float: The best equal error rate.
    """
    EER_Matrix = np.empty(len(fpr))
    for i in range(len(fpr)):
        eer = (fpr[i] + fnr[i]) / 2.0
        EER_Matrix[i] = eer
    best_eer = min(EER_Matrix)
    return best_eer


def calculate_threshold_bisectional(pos_cosine, neg_cosine, upper_threshold, lower_threshold, actual_labels):
    """
    Calculate the threshold using the bisectional method.

    Parameters:
    pos_cosine (list): The list of positive cosine values.
    neg_cosine (list): The list of negative cosine values.
    upper_threshold (float): The upper threshold value.
    lower_threshold (float): The lower threshold value.
    actual_labels (list): The list of actual labels.

    Returns:
    float: The calculated threshold.
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

    # print(lower_threshold, upper_threshold)
    while (abs(lower_threshold - upper_threshold) > 0.000002) or count == 1000:
        count = count + 1
        mid_threshold = (upper_threshold + lower_threshold) / 2
        mid_predicted_label, mid_calc_score = label_classifier(pos_cosine, neg_cosine, mid_threshold)

        cm_mid = metrics.confusion_matrix(actual_labels, mid_predicted_label)

        fpr_mid, fnr_mid = calculate_fpr_fnr(cm_mid)

        diff_mid = fpr_mid - fnr_mid

        if diff_mid > 0:
            lower_threshold = mid_threshold
        elif diff_mid < 0:
            upper_threshold = mid_threshold

    mid_threshold = (upper_threshold + lower_threshold) / 2
    return mid_threshold


def binary_search(low, mid, high):
    """
    Perform a binary search between low, mid, and high values.

    Parameters:
    low (float): The low value.
    mid (float): The mid value.
    high (float): The high value.

    Returns:
    tuple: A tuple containing the two closest values.
    """
    if abs(low - mid) < abs(high - mid):
        return low, mid
    else:
        return mid, high


def create_actual_labels(ratings):
    """
    Create actual labels based on ratings.

    Parameters:
    ratings (list): The list of ratings.

    Returns:
    list: The list of actual labels.
    """
    actual_labels = []
    for rating in ratings:
        if rating > 3:
            actual_labels.append(1)
        else:
            actual_labels.append(0)
    return actual_labels


def calculate_fpr_fnr(confusion_matrix_array):
    """
    Calculate the false positive rate and false negative rate from a confusion matrix.

    Parameters:
    confusion_matrix_array (numpy.ndarray): The confusion matrix.

    Returns:
    tuple: A tuple containing the false positive rate and false negative rate.
    """
    # Extract values from confusion matrix
    TN, FP, FN, TP = confusion_matrix_array.ravel()
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


def best_threshold(actual_labels, pos_cosine, neg_cosine, mid_threshold):
    """
    Calculate the best threshold for classification.

    Parameters:
    actual_labels (list): The list of actual labels.
    pos_cosine (list): The list of positive cosine values.
    neg_cosine (list): The list of negative cosine values.
    mid_threshold (float): The mid threshold value.

    Returns:
    tuple: A tuple containing the false acceptance rate and false rejection rate.
    """
    predicted_labels, calc_scores = label_classifier(pos_cosine, neg_cosine, mid_threshold)
    # calculate the Confusion Matrix
    cm = metrics.confusion_matrix(actual_labels, predicted_labels)
    # cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=[False, True])

    # cm_display.plot()
    # plt.show()
    FPR, FNR = calculate_fpr_fnr(cm)
    EER = (FPR + FNR) / 2

    true_acceptance = cm[1][1]  # True Acceptance (valid inputs accepted)
    false_rejection = cm[1][0]  # False Rejection (valid inputs rejected)
    false_acceptance = cm[0][1]  # False Acceptance (invalid inputs accepted)
    true_rejection = cm[0][0]  # True Rejection (invalid inputs rejected)

    print("The Threshold Used:", mid_threshold)
    print(f"Best Possible Confusion Matrix:\n{cm}")
    print(" Best Possible Equal Error Rate: ", EER)
    cm = rotate_2x2(cm)
    print(f"Rotated Possible Confusion Matrix:\n{cm}")

    total_valid_inputs = true_acceptance + false_rejection  # Total valid inputs
    total_invalid_inputs = false_acceptance + true_rejection  # Total invalid inputs

    far = false_acceptance / total_invalid_inputs
    frr = false_rejection / total_valid_inputs
    return far, frr


def rotate_2x2(matrix):
    """
    Rotate a 2x2 matrix by swapping elements diagonally.

    Parameters:
    matrix (numpy.ndarray): The 2x2 matrix.

    Returns:
    numpy.ndarray: The rotated matrix.
    """
    # Swap elements diagonally
    rotated_matrix = np.array([[matrix[1][1], matrix[1][0]],
                               [matrix[0][1], matrix[0][0]]])

    return rotated_matrix