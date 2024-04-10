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
import re


def calculate_metrics(actual_labels, positive_scores, negative_scores, threshold):
    # calculate predicted label and scores
    predicted_labels, calc_scores = label_classifier(positive_scores, negative_scores, threshold)
    # calculate the Confusion Matrix
    cm = metrics.confusion_matrix(actual_labels, predicted_labels)
    cm = rotate_2x2(cm)
    Uar = calculate_performance_matrix(cm, 0)
    fpr, fnr = calculate_fpr_fnr(cm)
    EER = (fpr + fnr) / 2

    return predicted_labels, calc_scores, cm, EER, fpr, fnr, Uar


def generate_freq(review, review_dict, gram):
    """
    Generate frequencies of words in positive and negative reviews.

    :param review: numpy array containing review data
    :param review_dict: dictionary containing word frequencies
    :param gram: U or B for Unigram or Bigram selection
    :return: tuple containing positive and negative word frequencies
    """
    neg_testing = review[review[:, 0] < 3]
    pos_testing = review[review[:, 0] > 3]

    pos_frequencies_test = generate_bag_of_words_frequencies(review_dict, pos_testing[:, 1], gram)
    pos_frequencies_test = np.array(pos_frequencies_test)
    pos_frequencies_test = pos_frequencies_test.reshape(-1, 1)

    neg_frequencies_test = generate_bag_of_words_frequencies(review_dict, neg_testing[:, 1], gram)
    neg_frequencies_test = np.array(neg_frequencies_test)
    neg_frequencies_test = neg_frequencies_test.reshape(-1, 1)
    return pos_frequencies_test, neg_frequencies_test


def generate_bigrams(text):

    words = text.split()
    bigrams = []
    for i in range(len(words) - 1):
        bigram = ' '.join([words[i], words[i + 1]])
        bigrams.append(bigram)
    return bigrams


def generate_bag_of_words_frequencies(dictionary, reviews, gram):
    """
    Generate bag of words frequencies from reviews.

    :param dictionary: dictionary containing word frequencies
    :param reviews: list of reviews
    :param gram: U or B for Unigram or Bigram selection
    :return: list of bag of words frequencies
    """
    dictionary_mapping = {word: index for index, word in enumerate(dictionary)}
    bag_words = np.zeros(len(dictionary))
    total_words = 0
    if gram == "U":
        for review_text in reviews:
            if isinstance(review_text, str) and review_text.lower() != 'nan':
                words = review_text.split()  # Split review text into words
                cleaned_words = [word.lower().strip(string.punctuation) for word in words]
                for word in cleaned_words:
                    index = dictionary_mapping.get(word)
                    if index is not None:
                        total_words += 1
                        bag_words[index] += 1
        return [(word_count / total_words) for word_count in bag_words] if total_words > 0 else np.zeros(
            len(dictionary))
    elif gram == "B":
        for review_text_temp in reviews:
            if isinstance(review_text_temp, str) and review_text_temp.lower() != 'nan':
                review_text_clean = review_text_temp.lower()
                review_text_clean = re.sub(r'\b\d+\b', '', review_text_clean)
                review_text_clean = re.sub(r'[^\w\s]', '', review_text_clean)
                words = generate_bigrams(review_text_clean)
                cleaned_words = [word.lower().strip(string.punctuation) for word in words]
                for word in cleaned_words:
                    index = dictionary_mapping.get(word)
                    if index is not None:
                        total_words += 1
                        bag_words[index] += 1

        return [(word_count / total_words) for word_count in bag_words] if total_words > 0 else np.zeros(len(dictionary))


def cosine_similarity_scores(all_frequencies):
    """
    Calculate cosine similarity scores.

    :param all_frequencies: array containing all word frequencies
    :return: tuple of positive and negative cosine similarity scores
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
    Classify labels based on positive and negative scores.

    :param positive_scores: array of positive scores
    :param negative_scores: array of negative scores
    :param threshold: threshold value for classification
    :return: tuple of labels and scores
    """
    labels = np.where((positive_scores - negative_scores - threshold) <= 0, 0, 1)
    scores = positive_scores - negative_scores
    return labels, scores


def label_classifier(pscore, nscore, threshold):
    """
    Classify labels based on positive and negative scores.

    :param pscore: array of positive scores
    :param nscore: array of negative scores
    :param threshold: threshold value for classification
    :return: tuple of labels and scores
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
    Calculate performance metrics.

    :param inputdata: input data
    :param flag: flag to print performance metrics
    :return: unweighted accuracy
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
    Calculate Detection Error Tradeoff (DET) curve.

    :param fpr: false positive rate
    :param fnr: false negative rate
    :return: None
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

    :param actual_labels: array of actual labels
    :param predicted_labels: array of predicted labels
    :return: tuple containing count of correct and wrong predictions
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
    Scheduler for confusion matrix calculation.

    :param actual_labels: array of actual labels
    :param positive_scores: array of positive scores
    :param negative_scores: array of negative scores
    :param thresholds: array of threshold values
    :return: tuple containing predicted labels, equal error rate, thresholds, confusion matrices,
             unweighted accuracy matrix, false positive rates, and false negative rates
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
    Find the maximum value in a list.

    :param list_of_values: list of values
    :return: index of the maximum value
    """
    max_value = max(list_of_values)
    max_index = list_of_values.index(max_value)

    return max_index


def plot_threshold_vs_accuracy(threshold_values, uar_values):
    """
    Plot threshold vs unweighted accuracy.

    :param threshold_values: array of threshold values
    :param uar_values: array of unweighted accuracy values
    :return: None
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
    Generate a classification report.

    :param actual_labels: array of actual labels
    :param predicted_labels: array of predicted labels
    :return: classification report
    """
    # Generate a classification report
    report = classification_report(actual_labels, predicted_labels, zero_division=1)
    return report


# Calculates the weight mask and returns it
def calculate_mask(negative_freq, positive_freq):
    """
    Calculate weight mask.

    :param negative_freq: array of negative frequencies
    :param positive_freq: array of positive frequencies
    :return: weight mask
    """
    delta = abs(positive_freq - negative_freq)
    summation = negative_freq + positive_freq
    modified_array = np.where(summation == 0, 1, summation)
    weights = delta / modified_array
    return weights


def calculate_filter(mask_temp, percent_filter):
    """
    Calculate filter.

    :param mask_temp: temporary mask
    :param percent_filter: percentage filter
    :return: tuple containing filtered mask
    """
    num_elements_to_zero = int(len(mask_temp) * percent_filter)
    # Sort the array and find the threshold value
    sorted_values = np.sort(mask_temp, axis=0)
    filter_threshold = sorted_values[num_elements_to_zero]
    # Set elements less than the threshold to zero
    mask_temp[mask_temp < filter_threshold] = 0
    return mask_temp


def squash(mask, n):
    return np.power(mask, n)


def calculate_EER(fpr, fnr):
    """
    Calculate Equal Error Rate (EER).

    :param fpr: false positive rate
    :param fnr: false negative rate
    :return: best EER value
    """
    EER_Matrix = np.empty(len(fpr))
    for i in range(len(fpr)):
        eer = (fpr[i] + fnr[i]) / 2.0
        EER_Matrix[i] = eer
    best_eer = min(EER_Matrix)
    return best_eer


def calculate_threshold_bisectional(pos_cosine, neg_cosine, upper_threshold, lower_threshold, actual_labels):
    """
    Calculate threshold using bisectional method.

    :param pos_cosine: positive cosine scores
    :param neg_cosine: negative cosine scores
    :param upper_threshold: upper threshold value
    :param lower_threshold: lower threshold value
    :param actual_labels: array of actual labels
    :return: threshold value
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
    Perform binary search.

    :param low: low value
    :param mid: mid value
    :param high: high value
    :return: tuple containing low and high values
    """
    if abs(low - mid) < abs(high - mid):
        return low, mid
    else:
        return mid, high


def create_actual_labels(ratings):
    """
    Create actual labels based on ratings.

    :param ratings: array of ratings
    :return: array of actual labels
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
    Calculate false positive rate and false negative rate.

    :param confusion_matrix_array: confusion matrix
    :return: tuple containing false positive rate and false negative rate
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
    Find the best threshold.

    :param actual_labels: array of actual labels
    :param pos_cosine: positive cosine scores
    :param neg_cosine: negative cosine scores
    :param mid_threshold: mid threshold value
    :return: tuple containing false acceptance rate and false rejection rate
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

def print_best_threshold(cm, EER, FPR, FNR, UAR, threshold):
    print("The Threshold Used:", threshold)
    print(f"Best Possible Confusion Matrix:\n{cm}")
    print("Best Possible FPR: ", FPR)
    print("Best Possible FNR: ", FNR)
    print("Best Possible Equal Error Rate: ", EER)
    print("Best Possible Unweighted Accuracy: ", UAR)



def rotate_2x2(matrix):
    """
    Rotate 2x2 matrix.

    :param matrix: input matrix
    :return: rotated matrix
    """
    # Swap elements diagonally
    rotated_matrix = np.array([[matrix[1][1], matrix[1][0]],
                               [matrix[0][1], matrix[0][0]]])

    return rotated_matrix


def print_commands():
    print("Commands:")
    print("M - Main function of the program")
    print("G - Graph generation")
    print("I - Input hyperparameters")
    print("P - Print values")
    print("E - Terminate program")
    print("F - Graph Filter")


def update_hyperparameters():
    print("Setting hyperparameters...")

    lower_threshold = float(input("Enter lower threshold value (Recommended value=-1): "))
    upper_threshold = float(input("Enter upper threshold value (Recommended value=1): "))
    lower_bound = float(input("Enter lower bound value (Recommended value=-0.1, an integer or float that's "
                              "less than zero): "))
    upper_bound = float(input("Enter upper bound value (Recommended value=0.1, an integer or float that's "
                              "greater than zero): "))
    step_value = float(input("Enter step value (Recommended value=0.02): "))
    threshold_values = np.arange(lower_bound, upper_bound, step_value)

    print("Hyperparameters set successfully...")
    return lower_threshold, upper_threshold, lower_bound, upper_bound, step_value, threshold_values


# Define the function for printing data
def print_data(hyperparameters, dictionary, pos_frequencies_train, neg_frequencies_train, pos_frequencies_test,
               neg_frequencies_test, lower_threshold, upper_threshold, EER, thresholds, confusion_matrices, report):
    print("Printing data...\n")

    # Print the hyperparameters
    print("Current parameters:")
    if hyperparameters:
        print(tabulate(hyperparameters, headers=["Parameter", "Value"], tablefmt="grid"))
    else:
        print("Hyperparameters are not set.")

    # Check the length of the dictionary
    print("\nLength of the dictionary:", len(dictionary))

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


def execute_main(reviews_matrix, dictionary, upper_threshold, lower_threshold, PERCENTAGE_TESTING, gram):
    print("Executing main function...")
    percentage_testing = PERCENTAGE_TESTING

    sorted_indices = np.argsort(reviews_matrix[0])
    reviews_matrix = reviews_matrix[:, sorted_indices]

    review_training, review_testing = train_test_split(reviews_matrix.T, test_size=percentage_testing,
                                                       random_state=42)
    neg_testing = review_testing[review_testing[:, 0] < 3]
    pos_testing = review_testing[review_testing[:, 0] > 3]
    pos_frequencies_test, neg_frequencies_test = generate_freq(review_training, dictionary, gram)
    combined_testing = np.concatenate((pos_testing, neg_testing), axis=0)
    print("this:", pos_frequencies_test.shape)
    all_frequencies_test = []

    run_num = 0

    for i in range(len(combined_testing)):
        run_num += 1
        review_text = combined_testing[i][1]
        if review_text == "":
            continue

        try:
            review_freq = generate_bag_of_words_frequencies(dictionary, [review_text], gram)
            # Append review_freq to the list
            all_frequencies_test.append(review_freq)
            print("Run " + str(run_num) + " successful.")
        except Exception as e:
            print("Run " + str(run_num) + " failed. Reason: " + str(e))

    all_frequencies_test = np.array(all_frequencies_test).T

    percent_filter = 0
    n = 6.5
    mask = calculate_mask(neg_frequencies_test, pos_frequencies_test)

    filtered_mask = calculate_filter(mask, percent_filter)
    filtered_mask = squash(filtered_mask, n)
    masked_neg = filtered_mask * neg_frequencies_test
    masked_pos = filtered_mask * pos_frequencies_test
    combined_matrix = np.concatenate((masked_pos, masked_neg, all_frequencies_test), axis=1)
    pos_cosine, neg_cosine = cosine_similarity_scores(combined_matrix)
    print(len(pos_cosine))
    print(all_frequencies_test.shape)
    actual_labels = create_actual_labels(combined_testing[:, 0])

    mid_threshold = calculate_threshold_bisectional(pos_cosine, neg_cosine, upper_threshold,
                                                    lower_threshold,
                                                    actual_labels)

    predicted_labels, calc_scores, cm, EER, fpr, fnr, Uar = calculate_metrics(actual_labels, pos_cosine,
                                                                              neg_cosine, mid_threshold)
    print_best_threshold(cm, EER, fpr, fnr, Uar, mid_threshold)
    report = statistics(actual_labels, predicted_labels)

    return report


def execute_uni_bi_main(reviews_matrix, dictionary, upper_threshold, lower_threshold, PERCENTAGE_TESTING, gram):
    print("Executing main function...")
    percentage_testing = PERCENTAGE_TESTING

    sorted_indices = np.argsort(reviews_matrix[0])
    reviews_matrix = reviews_matrix[:, sorted_indices]

    review_training, review_testing = train_test_split(reviews_matrix.T, test_size=percentage_testing,
                                                       random_state=42)
    neg_testing = review_testing[review_testing[:, 0] < 3]
    pos_testing = review_testing[review_testing[:, 0] > 3]
    pos_frequencies_test, neg_frequencies_test = generate_freq(review_training, dictionary, gram)
    combined_testing = np.concatenate((pos_testing, neg_testing), axis=0)
    print("this:", pos_frequencies_test.shape)
    all_frequencies_test = []

    run_num = 0

    # for i in range(len(combined_testing)):
    #     run_num += 1
    #     review_text = combined_testing[i][1]
    #     if review_text == "":
    #         continue
    #
    #     try:
    #         review_freq = generate_bag_of_words_frequencies(dictionary, [review_text], gram)
    #         # Append review_freq to the list
    #         all_frequencies_test.append(review_freq)
    #         print("Run " + str(run_num) + " successful.")
    #     except Exception as e:
    #         print("Run " + str(run_num) + " failed. Reason: " + str(e))

    all_frequencies_test = np.array(all_frequencies_test).T
    # np.savetxt(r'C:\Users\gdstren\Sentiment Graphs\Conference\combine_bi_all_freq.csv', all_frequencies_test, delimiter=",")
    # np.savetxt(r'C:\Users\gdstren\Sentiment Graphs\Conference\combine_bi_pos.csv', pos_frequencies_test, delimiter=",")
    # np.savetxt(r'C:\Users\gdstren\Sentiment Graphs\Conference\combine_bi_neg.csv', neg_frequencies_test, delimiter=",")

    i = 0
    w = 0
    EER_storage = []
    w_storage = []
    while i <= 2:
        combine_uni_all = np.loadtxt(r'C:\Users\gdstren\Sentiment Graphs\Conference\combine_uni_all_freq.csv', delimiter=",")
        combine_bi_all = np.loadtxt(r'C:\Users\gdstren\Sentiment Graphs\Conference\combine_bi_all_freq.csv',delimiter=",")
        combine_uni_pos = np.loadtxt(r'C:\Users\gdstren\Sentiment Graphs\Conference\combine_uni_pos.csv', delimiter=",")
        combine_bi_pos = np.loadtxt(r'C:\Users\gdstren\Sentiment Graphs\Conference\combine_bi_pos.csv', delimiter=",")
        combine_uni_neg = np.loadtxt(r'C:\Users\gdstren\Sentiment Graphs\Conference\combine_uni_neg.csv', delimiter=",")
        combine_bi_neg = np.loadtxt(r'C:\Users\gdstren\Sentiment Graphs\Conference\combine_bi_neg.csv', delimiter=",")

        combine_uni_all = combine_uni_all * (1-w)
        combine_uni_pos = combine_uni_pos * (1-w)
        combine_uni_neg = combine_uni_neg * (1-w)
        combine_bi_all = combine_bi_all * w
        combine_bi_pos = combine_bi_pos * w
        combine_bi_neg = combine_bi_neg * w

        merged_all = np.vstack([combine_uni_all, combine_bi_all])
        merged_pos = np.hstack([combine_uni_pos, combine_bi_pos])
        merged_neg = np.hstack([combine_uni_neg, combine_bi_neg])
        all_frequencies_test = merged_all
        neg_frequencies_test = np.reshape(merged_neg, (-1, 1))
        pos_frequencies_test = np.reshape(merged_pos, (-1, 1))



        percent_filter = 0
        n = 6.5
        mask = calculate_mask(neg_frequencies_test, pos_frequencies_test)

        filtered_mask = calculate_filter(mask, percent_filter)
        filtered_mask = squash(filtered_mask, n)
        masked_neg = filtered_mask * neg_frequencies_test
        masked_pos = filtered_mask * pos_frequencies_test
        combined_matrix = np.concatenate((masked_pos, masked_neg, all_frequencies_test), axis=1)
        pos_cosine, neg_cosine = cosine_similarity_scores(combined_matrix)

        actual_labels = create_actual_labels(combined_testing[:, 0])

        mid_threshold = calculate_threshold_bisectional(pos_cosine, neg_cosine, upper_threshold,
                                                        lower_threshold,
                                                        actual_labels)

        predicted_labels, calc_scores, cm, EER, fpr, fnr, Uar = calculate_metrics(actual_labels, pos_cosine,
                                                                                  neg_cosine, mid_threshold)
        EER_storage.append(EER)
        w_storage.append(w)
        print_best_threshold(cm, EER, fpr, fnr, Uar, mid_threshold)
        report = statistics(actual_labels, predicted_labels)
        w = w + 0.5
        i = i + 1
    print(EER_storage)
    print(w_storage)
    return report



def execute_filtered_main(review_matrix, dictionary, upper_threshold, lower_threshold, PERCENTAGE_THRESHOLD, gram):
    print("Executing filter function...")

    sorted_indices = np.argsort(review_matrix[0])
    reviews_matrix = review_matrix[:, sorted_indices]

    review_training, review_testing = train_test_split(reviews_matrix.T, test_size=PERCENTAGE_THRESHOLD,
                                                       random_state=42)
    neg_testing = review_testing[review_testing[:, 0] < 3]
    pos_testing = review_testing[review_testing[:, 0] > 3]

    pos_frequencies_test, neg_frequencies_test = generate_freq(review_training, dictionary, gram)

    combined_testing = np.concatenate((pos_testing, neg_testing), axis=0)

    all_frequencies_test = []

    run_num = 0

    for i in range(len(combined_testing)):
        run_num += 1
        review_text = combined_testing[i][1]
        if review_text == "":
            continue

        try:
            review_freq = generate_bag_of_words_frequencies(dictionary, [review_text], gram)
            # Append review_freq to the list
            all_frequencies_test.append(review_freq)
            print("Run " + str(run_num) + " successful.")
        except Exception as e:
            print("Run " + str(run_num) + " failed. Reason: " + str(e))

    all_frequencies_test = np.array(all_frequencies_test).T

    percent_filter = 0.24
    n = 1.75
    mask = calculate_mask(neg_frequencies_test, pos_frequencies_test)

    EER_array = []
    percent_filter_array = []
    while percent_filter < 0.4:
        print(percent_filter)
        percent_filter_array.append(percent_filter)

        filtered_mask = calculate_filter(mask, percent_filter)
        filtered_mask = squash(filtered_mask, n)
        masked_neg = filtered_mask * neg_frequencies_test
        masked_pos = filtered_mask * pos_frequencies_test
        combined_matrix = np.concatenate((masked_pos, masked_neg, all_frequencies_test), axis=1)
        pos_cosine, neg_cosine = cosine_similarity_scores(combined_matrix)

        actual_labels = create_actual_labels(combined_testing[:, 0])

        mid_threshold = calculate_threshold_bisectional(pos_cosine, neg_cosine, upper_threshold,
                                                        lower_threshold,
                                                        actual_labels)

        predicted_labels, calc_scores, cm, EER, fpr, fnr, Uar = calculate_metrics(actual_labels, pos_cosine,
                                                                                  neg_cosine, mid_threshold)
        print_best_threshold(cm, EER, fpr, fnr, Uar, mid_threshold)
        report = statistics(actual_labels, predicted_labels)
        print("EER: ", EER)
        EER_array.append(EER)
        percent_filter += 0.05


    # Plot the data
    plt.plot(percent_filter_array, EER_array, label='EER')
    # Add labels and title
    plt.xlabel('Percent Filter')
    plt.ylabel('EER')
    plt.title('Plot of EER over Filter')
    plt.legend()
    # Show grid
    plt.grid(True)

    # Show the plot
    plt.show()

    return report


def execute_filtered_squash_main(review_matrix, dictionary, upper_threshold, lower_threshold, PERCENTAGE_THRESHOLD, gram):
    print("Executing filter function...")

    sorted_indices = np.argsort(review_matrix[0])
    reviews_matrix = review_matrix[:, sorted_indices]

    review_training, review_testing = train_test_split(reviews_matrix.T, test_size=PERCENTAGE_THRESHOLD,
                                                       random_state=42)
    neg_testing = review_testing[review_testing[:, 0] < 3]
    pos_testing = review_testing[review_testing[:, 0] > 3]

    pos_frequencies_test, neg_frequencies_test = generate_freq(review_training, dictionary, gram)

    combined_testing = np.concatenate((pos_testing, neg_testing), axis=0)

    all_frequencies_test = []

    run_num = 0

    for i in range(len(combined_testing)):
        run_num += 1
        review_text = combined_testing[i][1]
        if review_text == "":
            continue

        try:
            review_freq = generate_bag_of_words_frequencies(dictionary, [review_text], gram)
            # Append review_freq to the list
            all_frequencies_test.append(review_freq)
            print("Run " + str(run_num) + " successful.")
        except Exception as e:
            print("Run " + str(run_num) + " failed. Reason: " + str(e))

    all_frequencies_test = np.array(all_frequencies_test).T

    actual_labels = create_actual_labels(combined_testing[:, 0])
    EER_array = []
    percent_filter_array = []
    n_array = []
    neg_zero_count_array = []
    pos_zero_count_array = []
    pf_limit = 0.5
    pf_ground = 0
    pf_increase = 0.01
    n_limit = 15
    n_ground = 0
    n_increase = 0.25
    total_iterations = round(
        ((n_limit - n_ground + n_increase) / n_increase) * ((pf_limit - pf_ground + pf_increase) / pf_increase))
    status_count = 0
    percent_filter = pf_ground

    while percent_filter <= pf_limit:
        n = n_ground

        while n <= n_limit:
            n_array.append(n)
            percent_filter_array.append(percent_filter)
            mask = calculate_mask(neg_frequencies_test, pos_frequencies_test)
            filtered_mask = calculate_filter(mask, percent_filter)
            filtered_mask = squash(filtered_mask, n)
            masked_neg = filtered_mask * neg_frequencies_test
            masked_pos = filtered_mask * pos_frequencies_test

            combined_matrix = np.concatenate((masked_pos, masked_neg, all_frequencies_test), axis=1)
            pos_cosine, neg_cosine = cosine_similarity_scores(combined_matrix)

            best_threshold = calculate_threshold_bisectional(pos_cosine, neg_cosine, upper_threshold,
                                                                                               lower_threshold,
                                                                                               actual_labels)

            predicted_labels, calc_scores, cm, EER, fpr, fnr, Uar = calculate_metrics(actual_labels,
                                                                                      pos_cosine,
                                                                                      neg_cosine,
                                                                                      best_threshold)
            print_best_threshold(cm, EER, fpr, fnr, Uar, best_threshold)

            report = statistics(actual_labels, predicted_labels)

            EER_array.append(EER)
            neg_zero_count_array.append(np.count_nonzero(masked_neg == 0))
            pos_zero_count_array.append(np.count_nonzero(masked_pos == 0))
            status_count += 1
            percent_complete = (status_count / total_iterations) * 100

            print(f'\r{status_count}/{total_iterations}  ({percent_complete:.2f}%)', end='', flush=True)

            n += n_increase

        percent_filter += pf_increase

    print("\nWriting to CSV...")

    # Define the file name
    file_name = r'C:\Users\gdstren\Sentiment Graphs\Conference\data.csv'

    if gram == "U":
        # Write data to CSV file
        with open(file_name, 'w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(['Percent_Filter', 'N', 'uni_EER'])  # Write header
            for i in range(len(percent_filter_array)):
                writer.writerow([percent_filter_array[i], n_array[i], EER_array[i]])

    elif gram == "B":
        # Read existing data from CSV file
        data = []
        with open(file_name, 'r') as file:
            reader = csv.reader(file)
            header = next(reader)
            data = [row for row in reader]

        # Add new column values
        for i, row in enumerate(data):
            row.append(EER_array[i])

        # Rewrite the CSV file with the updated data
        with open(file_name, 'w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(header + ['bi_EER'])
            writer.writerows(data)

    return report