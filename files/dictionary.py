import string


def generate_review_dictionary(dictionary, reviews):
    review_dict = []
    for review in reviews:
        words = review.split()
        for word in words:
            word_cleaned = word.lower().strip(string.punctuation)
            if word_cleaned in dictionary and word_cleaned not in review_dict:
                review_dict.append(word_cleaned)
    return sorted(review_dict)


def generate_bigrams(text):
    words = text.split()
    bigrams = []
    for i in range(len(words) - 1):
        bigram = ' '.join([words[i], words[i + 1]])
        bigrams.append(bigram)
    return bigrams