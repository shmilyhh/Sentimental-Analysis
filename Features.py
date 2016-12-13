from Evaluate import *

"""
Single word features, vinilla version
"""
def word_features(words):
    return dict([(word, True) for word in words])

print "Evaluate the single word features"
evaluate(word_features)

"""
High information features selection
use information gain
Information gain for classification is a measure of how common a feature is in a particular class 
compared to how common it is in all other classes
metrics for information gain is chi-square, which is included in BigramAssocMeasures class,
to use it we need to calculate the word overall frequency and frequency within each class,
FreqDist is used for overall frequecny of words, ConditionalFreqDist is used for frequecny with 
each class
BigramAssocMeasures.chi_sq(n_ii, (n_ix, n_xi), n_xx):
The arguments constitute the marginals of a contingency table, counting
the occurrences of particular events in a corpus. The letter i in the
suffix refers to the appearance of the word in question, while x indicates
the appearance of any word.
n_ii counts (w1, w2), i.e. the bigram being scored
n_ix counts (w1, *)
n_xi counts (*, w2)
n_xx counts (*, *), i.e. any bigram
     w1     ~w1
     ------ ------
 w2 | n_ii | n_xi | = n_xi
     ------ ------
~w2 | n_ix | n_xx |
     ------ ------
     = n_ix        TOTAL = n_xx
"""

word_featureDict = FreqDist()
label_word_featureDict = ConditionalFreqDist()

for word in movie_reviews.words(categories=['pos']):
    word_featureDict[word.lower()] += 1
    label_word_featureDict['pos'][word.lower()] += 1
for word in movie_reviews.words(categories=['neg']):
    word_featureDict[word.lower()] += 1
    label_word_featureDict['neg'][word.lower()] += 1

# calculate the n_ii, n_oi, n_io, n_oo in the BigramAssocMeasures.chi_sq()
# n_xi
pos_word_count = label_word_featureDict['pos'].N()
neg_word_count = label_word_featureDict['neg'].N()
# n_oo
total_word_count = pos_word_count + neg_word_count

# use BigramAssocMeasures.chi_sq() to calculate the score for each feature
word_scores = {}
for word, freq in word_featureDict.iteritems():
    pos_score = BigramAssocMeasures.chi_sq(label_word_featureDict['pos'][word], (freq, pos_word_count), total_word_count)
    neg_score = BigramAssocMeasures.chi_sq(label_word_featureDict['neg'][word], (freq, neg_word_count), total_word_count)
    word_scores[word] = pos_score + neg_score
# according to the scores to sort the features and get the top 10000 features
bestFeaturesDict = sorted(word_scores.iteritems(), key=lambda (w, s): s, reverse=True)[:10000]
bestFeatures = set([f for f, s in bestFeaturesDict])

def best_words_features(words):
    return dict([word, True] for word in words if word in bestFeatures)

print "Evaluate the top 10000 words features"
evaluate(best_words_features)

"""
Best bigram word features
Bigrams Collocations
BigramCollocationFinder maintains 2 internal FreqDists, one for individual word freq,
one for bigram freq. By using these freqs, it can score individual bigrams using a scoring function
provided by BigramAssocMeasures, such as Chi-Square. These score functions measure the collocation
correlation of 2 words, basically whether the bigram occurs about as frequently as each individual word.
Using nltk.util.bigrams to include all bigrams, the result only imporves a little, so including only 
significant features is better.
"""
def bigram_word_features(words, score_fn=BigramAssocMeasures.chi_sq, n=200):
    # use the BigramCollocationFinder.from_words() to extract all the bigrams in corpus
    bigrams = BigramCollocationFinder.from_words(words)
    # use the BigramAssocMeasures.chi_sq() score function to calculate the each bigram to get the valuable bigrams, then get top  n
    best_bigrams = bigrams.nbest(score_fn, n)
    bigram_featuresDict = dict([(bigram, True) for bigram in best_bigrams])
    return bigram_featuresDict

print "Evaluate top 200 bigrams features"
evaluate(bigram_word_features)

"""
combine the top 200 bigram features with the top 10000 best unigram features
"""
def uni_bi_gram_features(words):
    combined_featuresDict = bigram_word_features(words)
    combined_featuresDict.update(best_words_features(words))
    return combined_featuresDict
    
print "Evalute the comination of top 200 bigram features with the top 10000 best unigram features"
evaluate(uni_bi_gram_features)
    

