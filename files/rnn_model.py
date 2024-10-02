"""
This script implements a Recurrent Neural Network (RNN) for sentiment classification on Amazon product reviews.
It uses K-fold cross-validation, prints out key evaluation metrics, and displays plots of accuracy, loss, and confusion matrices.

### Workflow Breakdown:

1. **Data Loading and Preparation:**
    - The input is a CSV file containing reviews and ratings.
    - Rows with neutral ratings (i.e., 3-star ratings) are removed since they are considered neither positive nor negative.
    - The reviews (text) and ratings (labels) are extracted from the dataset.
    - Text reviews are preprocessed to ensure they are valid strings.

2. **Label Binarization:**
    - The ratings are converted into binary labels:
        - Ratings 1 and 2 are labeled as 0 (negative sentiment).
        - Ratings 4 and 5 are labeled as 1 (positive sentiment).

3. **Text Tokenization:**
    - The reviews are tokenized using a Tokenizer (from Keras), which converts words into numerical sequences.
    - Sequences are then padded to a uniform length (300 tokens per review) to ensure consistency in input data size.

4. **Dataset Splitting:**
    - The data is split into training (80%) and testing (20%) sets.
    - The training set will be used in cross-validation, while the test set is reserved for final evaluation.

5. **Model Definition:**
    - A simple RNN model is created:
        - **Embedding Layer**: Maps each token to a dense vector of fixed size (64).
        - **RNN Layer**: Processes the sequences with 64 units.
        - **Dropout Layer**: Reduces overfitting by randomly dropping some connections during training.
        - **Dense Layer**: Outputs a single value (0 or 1) for binary classification.
    - The model is compiled with the Adam optimizer and binary cross-entropy loss function.

6. **K-Fold Cross-Validation:**
    - **K-Fold (k=5)** splits the training set into 5 subsets, rotating through training and validation splits.
    - For each fold:
        - A subset is used for validation, while the remaining data is used for training.
        - The model is trained using the training data.
        - **Early Stopping** is used to prevent overfitting by stopping training if the validation loss stops improving.
        - **Learning Rate Reduction** adjusts the learning rate if no improvements are observed.
        - **Checkpoint** saves the best model for each fold.

7. **Evaluation for Each Fold:**
    - After training, the model is evaluated on the validation set using the following metrics:
        - **Accuracy**: The percentage of correct predictions.
        - **Precision**: How many of the positive predictions were actually positive.
        - **Recall**: How many of the actual positives were predicted correctly.
        - **F1 Score**: A balance between precision and recall.
        - **Confusion Matrix**: A table summarizing true positives, false positives, true negatives, and false negatives.
    - Confusion matrices and graphs for accuracy/loss are plotted for each fold.

8. **Average Validation Performance:**
    - After all folds are processed, the average validation accuracy across all folds is computed.

9. **Final Model Training:**
    - The final model is trained on the full training set (no validation split here).
    - **No early stopping** is applied during this stage since it's the final training on all the data.

10. **Test Set Evaluation:**
    - The final trained model is evaluated on the test set using the same metrics as in the validation phase:
        - **Accuracy**, **Precision**, **Recall**, **F1 Score**, and **Confusion Matrix** are printed for the test set.
    - The confusion matrix is plotted to visualize the model’s performance on the test data.

### Summary of Key Steps:
1. Load and clean data (remove neutral reviews).
2. Tokenize and pad reviews.
3. Split into training and test sets.
4. Perform K-Fold cross-validation to train and evaluate the model.
5. Display evaluation metrics for each fold.
6. Train the final model on the full training set.
7. Evaluate the final model on the test set and display metrics.
8. Plot graphs and confusion matrices for both validation and test sets.
"""


import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, KFold
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, SimpleRNN, Dense, Dropout, Bidirectional
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import accuracy_score
from sklearn.utils import class_weight
import seaborn as sns
import matplotlib.pyplot as plt


def run_rnn_model(data_filepath):
    """
    This function runs a Recurrent Neural Network (RNN) model for sentiment classification
    on the Amazon product reviews.

    Parameters:
    - data_filepath: str, path to the CSV file containing reviews and ratings.

    Returns:
    - test_accuracy: float, accuracy of the model on the test dataset.
    """
    # Load data
    data = pd.read_csv(data_filepath)

    # Remove rows with neutral (3) ratings
    data = data[data['Rating'] != 3]

    # Plot class distribution
    class_counts = data['Rating'].value_counts()
    print(f"Class distribution: {dict(class_counts)}")
    sns.barplot(x=class_counts.index, y=class_counts.values)
    plt.title("Class Distribution")
    plt.xlabel("Rating")
    plt.ylabel("Frequency")
    plt.show()

    # Extract reviews and ratings
    reviews = data['Review'].values
    ratings = data['Rating'].values

    # Clean and prepare the reviews (handle missing values)
    reviews = [str(review) if pd.notna(review) else "" for review in reviews]

    # Binarize the ratings (1, 2 -> 0, 4, 5 -> 1)
    labels = np.where(ratings <= 2, 0, 1)

    # Tokenize the text data
    tokenizer = Tokenizer(num_words=10000, oov_token='<OOV>')
    tokenizer.fit_on_texts(reviews)
    sequences = tokenizer.texts_to_sequences(reviews)

    # Pad sequences to ensure uniform length
    max_sequence_len = 300
    padded_sequences = pad_sequences(sequences, padding='post', maxlen=max_sequence_len)

    # Print tokenized sample
    print(f"Sample tokenized sequences: {sequences[:5]}")
    print(f"Sample padded sequences: {padded_sequences[:5]}")
    print(f"Sample labels: {labels[:5]}")

    # Split the dataset (80% train, 20% test)
    X_train, X_test, y_train, y_test = train_test_split(padded_sequences, labels, test_size=0.2, random_state=42)

    print(f"Total number of samples in training set: {len(X_train)}")
    print(f"Total number of samples in test set: {len(X_test)}")

    # Compute class weights to handle class imbalance
    class_weights_array = class_weight.compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
    class_weights = {i: class_weights_array[i] for i in range(len(class_weights_array))}

    def create_rnn_model():
        model = Sequential()
        model.add(Embedding(input_dim=10000, output_dim=64))
        model.add(Bidirectional(SimpleRNN(64, return_sequences=False)))  # Using a Bidirectional RNN
        model.add(Dropout(0.5))
        model.add(Dense(64, activation='relu'))  # Added an additional Dense layer with 64 units
        model.add(Dropout(0.5))
        model.add(Dense(1, activation='sigmoid'))  # Binary classification (positive/negative)

        # Using Adam optimizer with a learning rate of 0.001
        model.compile(optimizer=Adam(learning_rate=0.001), loss='binary_crossentropy', metrics=['accuracy'])
        return model

    # Perform k-fold cross-validation (k=5)
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    fold_no = 1
    accuracies = []

    for train_index, val_index in kf.split(X_train):
        print(f'\nTraining fold {fold_no}...')
        X_train_fold, X_val_fold = X_train[train_index], X_train[val_index]
        y_train_fold, y_val_fold = y_train[train_index], y_train[val_index]

        # Print the number of samples being processed in each fold
        print(f"Number of samples in training fold {fold_no}: {len(X_train_fold)}")
        print(f"Number of samples in validation fold {fold_no}: {len(X_val_fold)}")

        # Create and train the RNN model
        model = create_rnn_model()
        history = model.fit(X_train_fold, y_train_fold, epochs=10, batch_size=64, validation_data=(X_val_fold, y_val_fold), class_weight=class_weights)

        # Plot training and validation accuracy per epoch
        plt.plot(history.history['accuracy'], label='train_accuracy')
        plt.plot(history.history['val_accuracy'], label='val_accuracy')
        plt.title(f'Training and Validation Accuracy (Fold {fold_no})')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.show()

        # Evaluate the model on the validation data
        val_preds = model.predict(X_val_fold)
        val_preds = np.round(val_preds).flatten()  # Convert probabilities to binary 0/1
        accuracy = accuracy_score(y_val_fold, val_preds)
        print(f'Fold {fold_no} Accuracy: {accuracy * 100:.2f}%')
        accuracies.append(accuracy)
        fold_no += 1

    # Average accuracy across all folds
    print(f'\nAverage Validation Accuracy: {np.mean(accuracies) * 100:.2f}%')

    # Train final model on the entire training set and evaluate on test data
    final_model = create_rnn_model()
    final_model.fit(X_train, y_train, epochs=10, batch_size=64, class_weight=class_weights)

    # Evaluate on the test set
    test_preds = final_model.predict(X_test)
    test_preds = np.round(test_preds).flatten()
    test_accuracy = accuracy_score(y_test, test_preds)
    print(f'Test Accuracy: {test_accuracy * 100:.2f}%')

    return test_accuracy