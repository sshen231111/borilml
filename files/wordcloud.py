import numpy as np
from matplotlib import pyplot as plt
# import WordCloud


# def prepare_text_data(words, weights):
#     # Repeat each word according to its weight
#     return ' '.join(word for word, weight in zip(words, weights) for _ in range(weight))


def generate_synthetic_text(dictionary, normalized_weights, length_of_text):
    synthetic_text = []
    for word, weight in zip(dictionary, normalized_weights):
        word_count = int(weight * length_of_text) if (weight * length_of_text) % 1 <= 0.5 else int(weight * length_of_text) + 1
        synthetic_text.extend([word] * word_count)
    return synthetic_text



# def generate_word_cloud(text_data):
#     # Generate word cloud
#     all_text = ' '.join(all_text)
#     wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text_data)
#
#     # Display the generated word cloud
#     plt.figure(figsize=(10, 5))
#     plt.imshow(wordcloud, interpolation='bilinear')
#     plt.axis('off')
#     plt.show()
