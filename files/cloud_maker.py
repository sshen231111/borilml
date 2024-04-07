import pandas as pd
from wordcloud import WordCloud
import matplotlib.pyplot as plt

LENGTH_OF_TEXT = 100000


def generate_synthetic_text(dictionary, normalized_weights, length_of_text):
    """
    Generate synthetic text based on a dictionary of words and their normalized weights.

    :param dictionary: list of words
    :param normalized_weights: list of corresponding normalized weights
    :param length_of_text: desired length of synthetic text
    :return: synthetic text as a list of words
    """
    synthetic_text = []
    for word, weight in zip(dictionary, normalized_weights):
        word_count = int(weight * length_of_text) if (weight * length_of_text) % 1 <= 0.5 else int(
            weight * length_of_text) + 1
        synthetic_text.extend([word] * word_count)
    return synthetic_text


def generate_word_cloud_from_csv(csv_file):
    """
    Generate a word cloud from a CSV file containing word frequencies.

    :param csv_file: path to the CSV file containing word frequencies
    :return: None
    """
    # Read CSV file into a DataFrame
    df = pd.read_csv(csv_file, header=None, skiprows=1, names=['Word', 'Frequency'])

    # Convert DataFrame to a list of tuples (word, frequency)
    words_and_frequencies = list(zip(df['Word'], df['Frequency']))

    # Convert list of words and frequencies to a dictionary
    word_freq_dict = {word: freq for word, freq in words_and_frequencies if
                      (not word.startswith('a-') and not word.startswith('i-') and not word.startswith(
                          'is-') and not word.startswith('but-') and not word.startswith(
                          'because-') and not word.startswith('the-') and not word.startswith(
                          'and-') and not word.startswith('or-') and not word.startswith('be-') and not word.startswith(
                          'to-') and not word.startswith('of-') and not word.startswith('it-')) and not word.startswith(
                          'microwave-')
                      and
                      (not word.endswith('-a') and not word.endswith('-i') and not word.endswith(
                          '-is') and not word.endswith('-but') and not word.endswith('-because') and not word.endswith(
                          '-the') and not word.endswith(
                          '-and') and not word.endswith('-or') and not word.endswith('-be') and not word.endswith(
                          '-to') and not word.endswith('-of') and not word.endswith('-it') and not word.endswith(
                          'microwave'))}

    for word, freq in word_freq_dict.items():
        word_freq_dict[word] = round(1 / (freq / 404))
    # Generate word cloud
    wordcloud = WordCloud(width=800, height=400, background_color='white', max_words=100,
                          collocations=True).generate_from_frequencies(word_freq_dict)

    # Display the generated word cloud
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.show()


# generate_word_cloud_from_csv('../data/wordcloud/synthetic_bigrams_normalized.csv')
