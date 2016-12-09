import collections
import itertools
import nltk.classify.util
from nltk.corpus import stopwords
from nltk.metrics import precision, recall, f_measure
from nltk.classify import NaiveBayesClassifier
from nltk.corpus import movie_reviews
from nltk.collocations import BigramCollocationFinder
from nltk.metrics import BigramAssocMeasures
from nltk.probability import FreqDist, ConditionalFreqDist

def evaluate_classifier(featx):
    # get the negative label data
    negids = movie_reviews.fileids('neg')
    # get the positive label data
    posids = movie_reviews.fileids('pos')

    # The classifier training method expects to be given a list of tokens in the form of [(feats, label)], where 
    # the feats is a feature dictionary and label is the classification label.
    # For accuracy, use nltk.classify.util.accuracy
    negfeats = [(featx(movie_reviews.words(fileids=[f])), 'neg') for f in negids]
    posfeats = [(featx(movie_reviews.words(fileids=[f])), 'pos') for f in posids]

    # sample the corpus
    negcutoff = len(negfeats) * 3/4
    poscutoff = len(posfeats) * 3/4

    trainfeats = negfeats[:negcutoff] + posfeats[:poscutoff]
    testfeats = negfeats[negcutoff:] + posfeats[poscutoff:]
    print "train on %d instances, test on %d instances" %(len(trainfeats), len(testfeats))

    classifier = NaiveBayesClassifier.train(trainfeats)

    # In order to calculate the precision, recall and F, we need to build two sets for each classification label
    # a reference set of correct value; a test set of observed values
    refsets = collections.defaultdict(set)
    testsets = collections.defaultdict(set)

    for i, (feats, label) in enumerate(testfeats):
        refsets[label].add(i)
        observed = classifier.classify(feats)
        testsets[observed].add(i)
    
    print "accuracy: ", nltk.classify.util.accuracy(classifier, testfeats)
    print 'pos precision:', precision(refsets['pos'], testsets['pos'])
    print 'pos recall:', recall(refsets['pos'], testsets['pos'])
    print 'pos F-measure', f_measure(refsets['pos'], testsets['pos'])
    print 'neg precision:', precision(refsets['neg'], testsets['neg'])
    print 'neg recall:', recall(refsets['neg'], testsets['neg'])
    print 'neg F-measure', f_measure(refsets['neg'], testsets['neg'])
    classifier.show_most_informative_features()
    
    # test
    review = "This movie is not insulting"
    review_features = featx(review.split())
    print classifier.classify(review_features)

"""
Single word features, vinilla version
"""
# NLTK works with featstructs, which can be simple dictionaries mapping a feature name to feature value
def word_feats(words):
    return dict([(word, True) for word in words])

# evaluate the single word featrues
print "evaluating single word features"
evaluate_classifier(word_feats)

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
 w2 | n_ii | n_oi | = n_xi
     ------ ------
~w2 | n_io | n_oo |
     ------ ------
     = n_ix        TOTAL = n_xx
"""

word_fd = FreqDist()
label_word_fd = ConditionalFreqDist()

for word in movie_reviews.words(categories=['pos']):
    word_fd[word.lower()] += 1
    label_word_fd['pos'][word.lower()] += 1
    
for word in movie_reviews.words(categories=['neg']):
    word_fd[word.lower()] += 1
    label_word_fd['neg'][word.lower()] += 1

# number of the element in one category    
pos_word_count = label_word_fd['pos'].N()
neg_word_count = label_word_fd['neg'].N()
total_word_count = pos_word_count + neg_word_count

# calculate each word score
word_scores = {}
for word, freq in word_fd.iteritems():
    pos_score = BigramAssocMeasures.chi_sq(label_word_fd['pos'][word], (freq, pos_word_count), total_word_count)
    neg_score = BigramAssocMeasures.chi_sq(label_word_fd['neg'][word], (freq, neg_word_count), total_word_count)
    word_scores[word] = pos_score + neg_score
    
best = sorted(word_scores.iteritems(), key=lambda (w, s): s, reverse=True)[:10000]
bestwords = set([w for w, s in best])

def best_word_feats(words):
    return dict([word, True] for word in words if word in bestwords)

print "evaluating best words features"
evaluate_classifier(best_word_feats)

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
def bigram_word_feats(words, score_fn=BigramAssocMeasures.chi_sq, n=200):
    bigram_finder = BigramCollocationFinder.from_words(words)
    bigrams = bigram_finder.nbest(score_fn, n)
    d = dict([(bigram, True) for bigram in bigrams])
    d.update(best_word_feats(words))
    return d

print "evaluating best words + bigrams chi_square word features"
evaluate_classifier(bigram_word_feats)