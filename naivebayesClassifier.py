import nltk.classify.util
from nltk.classify import NaiveBayesClassifier
from nltk.corpus import movie_reviews

# NLTK works with featstructs, which can be simple dictionaries mapping a feature name to feature value
def word_feats(words):
    return dict([(word, True) for word in words])
    
# get the negative label data
negids = movie_reviews.fileids('neg')
# get the positive label data
posids = movie_reviews.fileids('pos')

# The classifier training method expects to be given a list of tokens in the form of [(feats, label)], where 
# the feats is a feature dictionary and label is the classification label.
# For accuracy, use nltk.classify.util.accuracy
negfeats = [(word_feats(movie_reviews.words(fileids=[f])), 'negative') for f in negids]
posfeats = [(word_feats(movie_reviews.words(fileids=[f])), 'positive') for f in posids]

# sample the corpus
negcutoff = len(negfeats) * 3/4
poscutoff = len(posfeats) * 3/4

trainfeats = negfeats[:negcutoff] + posfeats[:poscutoff]
testfeats = negfeats[negcutoff:] + posfeats[poscutoff:]
print "train on %d instances, test on %d instances" %(len(trainfeats), len(testfeats))

classifier = NaiveBayesClassifier.train(trainfeats)
print "accuracy: ", nltk.classify.util.accuracy(classifier, testfeats)
classifier.show_most_informative_features()


# test
review = "This movie is not bad"
review_features = word_feats(review.split())
print classifier.classify(review_features)
