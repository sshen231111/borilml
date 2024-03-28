import numpy as np
from sklearn.model_selection import train_test_split
from tabulate import tabulate
import csv_manipulation
import classifier
import wordcloud

# percentage_testing = 0.1
# HYPERPARAMETERS = {
#
# }


def main():
    # Hyperparameters
    percentage_testing = 0.1
    lower_threshold = None
    upper_threshold = None
    lower_bound = None
    upper_bound = None
    step_value = None

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
    length_of_text = 214785

    # Load data
    reviews_matrix, review_dict = csv_manipulation.load_data()

    while True:
        if not parameters_set:
            print("Setting hyperparameters...")
            default = input("Enter Y for default settings): ")
            if default == "Y":
                lower_threshold = -10
                upper_threshold = 10
                lower_bound = -0.1
                upper_bound = 0.1
                step_value = 0.02
            else:
                lower_threshold = float(input("Enter lower threshold value (Recommended value=-1): "))
                upper_threshold = float(input("Enter upper threshold value (Recommended value=1): "))
                lower_bound = float(input("Enter lower bound value (Recommended value=-0.1, an integer or float that's "
                                          "less than zero): "))
                upper_bound = float(input("Enter upper bound value (Recommended value=0.1, an integer or float that's "
                                          "greater than zero): "))
                step_value = float(input("Enter step value (Recommended value=0.02): "))
                threshold_values = np.arange(lower_bound, upper_bound, step_value)
            percentage_training = 1 - percentage_testing

            # Store hyperparameters in a list of tuples
            hyperparameters = [
                ("Percentage for testing data", percentage_testing),
                ("Percentage for training data", percentage_training),
                ("Lower threshold value", lower_threshold),
                ("Upper threshold value", upper_threshold),
                ("Lower bound value", lower_bound),
                ("Upper bound value", upper_bound),
                ("Step value", step_value)
            ]

            parameters_set = True

        print("Press 'H' for a list of commands.")
        command = input("Enter a command: ").strip().upper()

        if command == 'H':
            print("Commands:")
            print("M - Main function of the program")
            print("G - Graph generation")
            print("I - Input hyperparameters")
            print("P - Print values")
            print("E - Terminate program")

        elif command == 'M':
            print("Executing main function...")

            if (percentage_testing is not None and lower_threshold is not None and upper_threshold is not None
                    and lower_bound is not None and upper_bound is not None and step_value is not None):

                sorted_indices = np.argsort(reviews_matrix[0])
                reviews_matrix = reviews_matrix[:, sorted_indices]

                review_training, review_testing = train_test_split(reviews_matrix.T, test_size=percentage_testing,
                                                                   random_state=42)
                # These variables contain the respective positive and negative reviews with their reviews
                neg_testing = review_testing[review_testing[:, 0] < 3]
                pos_testing = review_testing[review_testing[:, 0] > 3]

                # These variables contain the frequencies of the positive and negative bags of words
                pos_frequencies_test, neg_frequencies_test = classifier.generate_freq(review_training, review_dict)

                # Combined_testing contains all the scores and reviews
                combined_testing = np.concatenate((pos_testing, neg_testing), axis=0)

                all_frequencies_test = []

                for i in range(len(combined_testing)):
                    review_rate = combined_testing[i][0]
                    review_text = combined_testing[i][1]
                    if review_text == "":
                        continue

                    try:
                        review_freq = classifier.generate_bag_of_words_frequencies(review_dict, [review_text])
                        # Append review_freq to the list
                        all_frequencies_test.append(review_freq)
                    except:
                        print(review_text)
                # Contains the dictionary frequencies of words
                all_frequencies_test = np.array(all_frequencies_test).T
                actual_labels = classifier.create_actual_labels(combined_testing[:, 0])
                # save_to_csv(all_frequencies_test, "all_freq_test_unigram")
                # save_to_csv(actual_labels, "actual_labels_unigram")
                # if .46 is entered then Any values in the mask array below the 23rd percentile would be set to 0.
                # Any values above the 77th percentile would also be set to 0.
                # In essence, this function would retain only the values within the middle 54% of the data range, setting the lowest 23% and the highest 23% of values to 0.
                percent_filter = 0
                n = 7
                # percent_filter_array = np.arange(0.4, 0.8, 0.01)
                mask = classifier.calculate_mask(neg_frequencies_test, pos_frequencies_test)
                normalized_mask = classifier.normalize_mask(mask)
                # print(normalized_mask[:400])
                synthetic_text = wordcloud.generate_synthetic_text(review_dict, normalized_mask, length_of_text)
                csv_manipulation.save_synthetic_text_to_csv(synthetic_text, "../data/synthetic_normalized.csv")

                mask_squashed = classifier.squash(mask, n)
                normalized_mask_squashed = classifier.normalize_mask(mask_squashed)
                synthetic_text_two = wordcloud.generate_synthetic_text(review_dict, normalized_mask_squashed, length_of_text)
                csv_manipulation.save_synthetic_text_to_csv(synthetic_text_two, "../data/synthetic_normalized_squashed.csv")

                print("The non zero values in mask:", classifier.non_zero_values(neg_frequencies_test))
                print("The non zero values in mask:", classifier.non_zero_values(pos_frequencies_test))
                # 11,774 Total
                # Start with 8559 nonzero terms- 73% are nonzero terms
                # EER Diverges around 6501 nonzero terms- 55% are nonzero terms
                mask = classifier.squash(mask, n)
                # save_to_csv(mask, "squashed_unigram_mask")
                filtered_mask = classifier.calculate_filter(mask, percent_filter)
                masked_neg = filtered_mask * neg_frequencies_test
                masked_pos = filtered_mask * pos_frequencies_test

                combined_matrix = np.concatenate((masked_pos, masked_neg, all_frequencies_test), axis=1)
                # combined_matrix = np.concatenate((pos_frequencies_test, neg_frequencies_test, all_frequencies_test), axis=1)
                pos_cosine, neg_cosine = classifier.cosine_similarity_scores(combined_matrix)

                upper_threshold, lower_threshold, best_threshold = classifier.calculate_threshold_bisectional(
                    pos_cosine,
                    neg_cosine,
                    upper_threshold,
                    lower_threshold,
                    actual_labels)
                # confusion_matrices, uar_matrix, all_fpr, all_fnr, all_EER, all_neg, all_pos = (
                #     percent_filter_calc(actual_labels, pos_frequencies_test, neg_frequencies_test,percent_filter_array, mask, all_frequencies_test, 100, -100))

                # threshold_matrix = np.arange(best_threshold - 0.5, best_threshold + 0.5, 0.002).reshape(-1, 1)

                predicted_labels, calc_scores, cm, EER, fpr, fnr, Uar = classifier.calculate_metrics(actual_labels,
                                                                                                     pos_cosine,
                                                                                                     neg_cosine,
                                                                                                     best_threshold)
                classifier.print_best_threshold(cm, EER, fpr, fnr, Uar, best_threshold)

                # plot_metrics(percent_filter_array, all_fpr, all_fnr, all_EER)
                # plot_nonzero(percent_filter_array, all_neg, all_pos)

                report = classifier.statistics(actual_labels, predicted_labels)
                print("The non zero values in mask:", classifier.non_zero_values(mask))

            else:
                print("Hyperparameters are not set. Please set hyperparameters first.")

        elif command == 'G':
            print("Generating graphs...")

            if thresholds is not None and uar_matrix is not None and all_fpr is not None and all_fnr is not None:
                classifier.plot_threshold_vs_accuracy(thresholds, uar_matrix)
                classifier.calculate_det(all_fpr, all_fnr)
            else:
                print("Graphs cannot be generated. Required data is missing.")

        elif command == 'I':
            print("Setting hyperparameters...")

            percentage_testing = float(input("Enter percentage for testing data (e.g., 0.1 for 10%): "))
            lower_threshold = float(input("Enter lower threshold value (Recommended value=-1): "))
            upper_threshold = float(input("Enter upper threshold value (Recommended value=1): "))
            lower_bound = float(input("Enter lower bound value (Recommended value=-0.1, an integer or float that's "
                                      "less than zero): "))
            upper_bound = float(input("Enter upper bound value (Recommended value=0.1, an integer or float that's "
                                      "greater than zero): "))
            step_value = float(input("Enter step value (Recommended value=0.02): "))
            threshold_values = np.arange(lower_bound, upper_bound, step_value)
            percentage_training = 1 - percentage_testing

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
                    max_index = classifier.find_max(uar_matrix)
                    print("\nThe peak results are:\n")
                    print("Threshold:", thresholds[max_index])
                    print("Confusion Matrix:")
                    headers = ["", "Expected Positive", "Expected Negative"]
                    matrix_data = [["", "P", "N"],
                                   ["P", confusion_matrices[max_index][1, 1], confusion_matrices[max_index][1, 0]],
                                   ["N", confusion_matrices[max_index][0, 1], confusion_matrices[max_index][0, 0]]]
                    print(tabulate(matrix_data, headers=headers, tablefmt="grid"))
                    classifier.calculate_performance_matrix(confusion_matrices[max_index], 1)
                else:
                    print("\nThe peak results are:\n")
                    print("Threshold:", thresholds[0])
                    print("Confusion Matrix:")
                    headers = ["", "Expected Positive", "Expected Negative"]
                    matrix_data = [["", "P", "N"],
                                   ["P", confusion_matrices[0][1, 1], confusion_matrices[0][1, 0]],
                                   ["N", confusion_matrices[0][0, 1], confusion_matrices[0][0, 0]]]
                    print(tabulate(matrix_data, headers=headers, tablefmt="grid"))
                    classifier.calculate_performance_matrix(confusion_matrices[0], 1)
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
