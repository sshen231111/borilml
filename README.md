# borilml
An overview of our project is to classifier Amazon product reviews based on the text used
within. The research started with designing a web scraper to retrieve reviews on Amazon
products. We were able to scrape 18,487 reviews. We skipped 3-star reviews since our
classifier only determines if the reviews are negative or positive. Once the reviews were
successfully scraped, the dataset was split into training and test sets. The training set was
then used to generate a dictionary of every word used in the reviews, discarding repeats.
The training reviews are then broken into negative (1 and 2 stars) and positive (4 and 5
stars) groups. Using the two new groups, a positive and negative bag of words are created
from referencing the dictionary and counting the number of times the positive reviews use
each word in the dictionary and then doing the same for the negative. After these counts
are made, they are then normalized to create frequencies of word usage. The frequencies
are then used to develop a mask and filter to improve the results. The mask is designed by
taking the absolute value of the difference between the positive and negative frequencies
and dividing it by the summation of two frequencies. This mask is then multiplied by the
original positive and negative frequencies. The mask was created to balance the words,
since some words are unrepresented due to not being used a lot but are mostly only used
in one type of review. The filter is then designed off the mask since it can be applied to
filter out words that were used roughly the same number of times in both positive and
negative reviews. Ideally this will filter out words that are conjunctions, pronouns, and
prepositions. After this filter is applied, testing data that was set aside can be used. The
testing data will then take each individual review, compare it to the dictionary designed
earlier to create its own bag of words. The bag of words will then be compared to both the
negative and positive bag of words designed from the testing set. The comparison is done
by using cosine distancing and then the review will be classified based on which cosine
distance is smaller. This process is done for each individual review. The original star
reviews are still stored within the system, so that is compared to the classifier to determine
if it was correct or not. Our system then will output its overall unweighted accuracy, EER, F1
score, confusion matrix, and precision. Using the EER, an optimal threshold is applied to
increase the accuracy of the system.
