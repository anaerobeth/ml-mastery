# Identify gender from first names
# Adapted from https://www.nltk.org/book/ch06.html

import random
import nltk
from nltk.corpus import names

def gender_features(word):
    return {'last_letter': word[-1]}

male_names = [(name, 'male') for name in names.words('male.txt')]
female_names = [(name, 'female') for name in names.words('female.txt')]
labeled_names = male_names + female_names
random.shuffle(labeled_names)

features = [(gender_features(n), gender) for (n, gender) in labeled_names]
train_set, test_set = features[500:], features[:500]

# 1. Naive Bayes + last letter
clf = nltk.NaiveBayesClassifier.train(train_set)
print(nltk.classify.accuracy(clf, test_set))
# 0.79
print(clf.show_most_informative_features(3))
# Most Informative Features
#            last_letter = 'a'            female : male   =     36.8 : 1.0
#            last_letter = 'k'              male : female =     31.3 : 1.0
#            last_letter = 'f'              male : female =     26.6 : 1.0

# Starks
print(clf.classify(gender_features('Eddard'))) # male
print(clf.classify(gender_features('Sansa'))) # female
# Targaryens
print(clf.classify(gender_features('Rhaegar'))) # male
print(clf.classify(gender_features('Daenerys'))) # male - wrong
# Lannisters
print(clf.classify(gender_features('Tyrion'))) # male
print(clf.classify(gender_features('Cersei'))) # female

# Others
print(clf.classify(gender_features('Melisandre'))) # female
print(clf.classify(gender_features('Tormund'))) # male

