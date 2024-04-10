import pandas as pd
from wordcloud import WordCloud
import matplotlib.pyplot as plt

LENGTH_OF_TEXT = 100000


def generate_synthetic_text(dictionary, normalized_weights, length_of_text):
    synthetic_text = []
    for word, weight in zip(dictionary, normalized_weights):
        word_count = int(weight * length_of_text) if (weight * length_of_text) % 1 <= 0.5 else int(
            weight * length_of_text) + 1
        synthetic_text.extend([word] * word_count)
    return synthetic_text


def generate_word_cloud_from_csv(csv_file, blacklist_file):
    df = pd.read_csv(csv_file, header=None, skiprows=1, names=['Word', 'Frequency'])
    words_and_frequencies = list(zip(df['Word'], df['Frequency']))

    with open(blacklist_file, 'r') as f:
        blacklisted_bigrams = f.read().splitlines()

    word_freq_dict = {word: freq for word, freq in words_and_frequencies if
                      not any(bigram in word for bigram in blacklisted_bigrams)}

    for word, freq in word_freq_dict.items():
        word_freq_dict[word] = round(1 / (freq / 404))

    wordcloud = WordCloud(width=800, height=400, background_color='white', max_words=100,
                          collocations=True).generate_from_frequencies(word_freq_dict)

    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.show()


generate_word_cloud_from_csv('../data/wordcloud/synthetic_bigrams_normalized.csv',
                             '../data/wordcloud/blacklisted_bigrams.txt')
