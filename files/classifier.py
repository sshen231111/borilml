import re
import string

import numpy as np
from matplotlib import pyplot as plt
from sklearn import metrics
from sklearn.metrics import DetCurveDisplay, classification_report
from sklearn.metrics.pairwise import cosine_similarity

from files.dictionary import generate_bigrams


def generate_bag_of_words_frequencies(dictionary, reviews):
    dictionary_mapping = {word: index for index, word in enumerate(dictionary)}
    bag_words = np.zeros(len(dictionary))
    total_words = 0
    for review_text_temp in reviews:
        # print(review_text_temp)
        if isinstance(review_text_temp, str) and review_text_temp.lower() != 'nan':
            review_text_clean = review_text_temp.lower()
            review_text_clean = re.sub(r'\b\d+\b', '', review_text_clean)
            review_text_clean = re.sub(r'[^\w\s]', '', review_text_clean)
            # print(review_text_clean)
            words = review_text_clean.split()
            # words = generate_bigrams(review_text_clean)
            # print(words)
            cleaned_words = [word.lower().strip(string.punctuation) for word in words]
            for word in cleaned_words:
                # print(word)
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


def percent_filter_calc(actual_labels, pos_frequencies_test, neg_frequencies_test, percent_filter_array, mask,
                        all_frequencies_test, upper_threshold, lower_threshold):
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


def plot_metrics(percent_filter_array, all_fpr, all_fnr, all_EER):
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


def find_max(list):
    max_value = max(list)
    max_index = list.index(max_value)

    return max_index


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


def normalize_mask(weights):
    total_weight = np.sum(weights)
    normalized_mask = weights / total_weight
    return normalized_mask


def calculate_filter(mask, percent_filter):
    sorted_mask = np.sort(mask)
    percent_filter = np.percentile(sorted_mask, (percent_filter / 2) * 100)
    inverse_filter = np.percentile(sorted_mask, (100 - ((percent_filter / 2) * 100)))

    mask[mask < percent_filter] = 0
    mask[mask > inverse_filter] = 0
    return mask


def calculate_filter_ends(mask, percent_filter):
    upper_percent = (percent_filter * 3) / 4
    lower_percent = percent_filter / 4

    upper_percent = np.percentile(mask, upper_percent * 100)
    lower_percent = np.percentile(mask, lower_percent * 100)

    mask[mask < lower_percent] = 0
    mask[mask > upper_percent] = 0
    return mask


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

    while count < 1000:
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
    if abs(low - mid) < abs(high - mid):
        return low, mid
    else:
        return mid, high


def calculate_fpr_fnr(confusion_matrix):
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


def rotate_2x2(matrix):
    # Swap elements diagonally
    rotated_matrix = np.array([[matrix[1][1], matrix[1][0]],
                               [matrix[0][1], matrix[0][0]]])
    return rotated_matrix


def non_zero_values(frequencies):
    num_non_zero = np.count_nonzero(frequencies)
    # print("Non Zero Frequencies: ", num_non_zero)
    return num_non_zero


def squash(mask, n):
    return np.power(mask, n)


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


def create_actual_labels(ratings):
    actual_labels = []
    for rating in ratings:
        if rating > 3:
            actual_labels.append(1)
        else:
            actual_labels.append(0)
    return actual_labels


def print_best_threshold(cm, EER, FPR, FNR, UAR, threshold):
    print("The Threshold Used:", threshold)
    print(f"Best Possible Confusion Matrix:\n{cm}")
    print("Best Possible FPR: ", FPR)
    print("Best Possible FNR: ", FNR)
    print("Best Possible Equal Error Rate: ", EER)
    print("Best Possible Unweighted Accuracy: ", UAR)


def plot_threshold_vs_accuracy(threshold_values, uar_values):
    # Plotting the threshold_matrix(x axis) vs unweighted accuracy (UAR)
    plt.figure(figsize=(8, 5))
    plt.plot(threshold_values, uar_values, marker='o', linestyle='-')
    plt.title('Threshold vs Unweighted Accuracy')
    plt.xlabel('Threshold')
    plt.ylabel('Unweighted Accuracy (UAR)')
    plt.grid(True)
    plt.show()
