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
    labels = np.where((positive_scores - negative_scores - threshold) <= 0, 0, 1)
    scores = positive_scores - negative_scores
    return labels, scores


def label_classifier(pscore, nscore, threshold):
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
    display = DetCurveDisplay(fpr=fpr, fnr=fnr, estimator_name="DET Curve")
    display.plot()
    plt.xlabel('False Positive Rate (FPR)')
    plt.ylabel('False Negative Rate (FNR)')
    plt.title('Detection Error Tradeoff (DET) Curve')
    plt.show()


def count_correct_predictions(actual_labels, predicted_labels):
    correct_count = 0
    wrong_count = 0
    for actual, predicted in zip(actual_labels, predicted_labels):
        if actual == predicted:
            correct_count += 1
        else:
            wrong_count += 1
    return correct_count, wrong_count


def confusion_matrix_scheduler(actual_labels, positive_scores, negative_scores, thresholds):
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
    max_value = max(list_of_values)
    max_index = list_of_values.index(max_value)

    return max_index


def plot_threshold_vs_accuracy(threshold_values, uar_values):
    # Plotting the threshold_matrix(x axis) vs unweighted accuracy (UAR)
    plt.figure(figsize=(8, 5))
    plt.plot(threshold_values, uar_values, marker='o', linestyle='-')
    plt.title('Threshold vs Unweighted Accuracy')
    plt.xlabel('Threshold')
    plt.ylabel('Unweighted Accuracy (UAR)')
    plt.grid(True)
    plt.show()


def statistics(actual_labels, predicted_labels):
    # Generate a classification report
    report = classification_report(actual_labels, predicted_labels, zero_division=1)
    return report


# Calculates the weight mask and returns it
def calculate_mask(negative_freq, positive_freq):
    delta = abs(positive_freq - negative_freq)
    summation = negative_freq + positive_freq
    modified_array = np.where(summation == 0, 1, summation)
    weights = delta / modified_array
    return weights


def calculate_filter(mask_temp, percent_filter):
    num_elements_to_zero = int(len(mask_temp) * percent_filter)
    # Sort the array and find the threshold value
    sorted_values = np.sort(mask_temp, axis=0)
    filter_threshold = sorted_values[num_elements_to_zero]
    # Set elements less than the threshold to zero
    mask_temp[mask_temp < filter_threshold] = 0
    return mask_temp, num_elements_to_zero


def calculate_EER(fpr, fnr):
    EER_Matrix = np.empty(len(fpr))
    for i in range(len(fpr)):
        eer = (fpr[i] + fnr[i]) / 2.0
        EER_Matrix[i] = eer
    best_eer = min(EER_Matrix)
    return best_eer


def calculate_threshold_bisectional(pos_cosine, neg_cosine, upper_threshold, lower_threshold, actual_labels):
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
    if abs(low - mid) < abs(high - mid):
        return low, mid
    else:
        return mid, high


def create_actual_labels(ratings):
    actual_labels = []
    for rating in ratings:
        if rating > 3:
            actual_labels.append(1)
        else:
            actual_labels.append(0)
    return actual_labels


def calculate_fpr_fnr(confusion_matrix_array):
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
    # Swap elements diagonally
    rotated_matrix = np.array([[matrix[1][1], matrix[1][0]],
                               [matrix[0][1], matrix[0][0]]])

    return rotated_matrix