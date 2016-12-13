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

def evaluate(feature_method):
    # get the file id from the nltk corpus
    negativeId = movie_reviews.fileids('neg')
    positiveId = movie_reviews.fileids('pos')
     
    # use the feature_method to get the features that are used in the naive bayes classifier
    # the format for the classifier is [(features, label)]
    negFeatures = [(feature_method(movie_reviews.words(fileids=[id])), 'neg') for id in negativeId]
    posFeatures = [(feature_method(movie_reviews.words(fileids=[id])), 'pos') for id in positiveId]
     
    # split the corpus into training data and testing data
    negCutoff = len(negFeatures) * 3/4
    posCutoff = len(posFeatures) * 3/4
     
    trainFeatures = negFeatures[:negCutoff] + posFeatures[:posCutoff]
    testFeatures = negFeatures[negCutoff:] + posFeatures[posCutoff:]
    print "the number of the training features: ", len(trainFeatures)
    print "the number of the testing features: ", len(testFeatures)
     
    # get the classifier from the nltk
    classifier = NaiveBayesClassifier.train(trainFeatures)
     
    # calculate the recall and precision and F-measure
    # By using the functions, we need build two set: set of correct set and set of observed set
    correctSets = collections.defaultdict(set)   
    observedSets = collections.defaultdict(set)
     
    for i, (f, l) in enumerate(testFeatures):
        correctSets[l].add(i)
        observed = classifier.classify(f)
        observedSets[observed].add(i)
    
    print "Accuracy: ", nltk.classify.util.accuracy(classifier, testFeatures)
    print "Recall: "
    print "\tpos: ", recall(correctSets['pos'], observedSets['pos']), "\tneg: ", recall(correctSets['neg'], observedSets['neg'])
    print "Precision: "
    print "\tpos: ", precision(correctSets['pos'], observedSets['pos']), "\tneg: ", precision(correctSets['neg'], observedSets['neg'])
    print "F-Measure: "
    print "\tpos", f_measure(correctSets['pos'], observedSets['pos']), "\tneg: ", f_measure(correctSets['neg'], observedSets['neg'])
    
    classifier.show_most_informative_features()
    
    # Test
    review = "This is movie is not insulting"
    review_features = feature_method(review.split())
    print classifier.classify(review_features)
    print