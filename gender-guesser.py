# How many of the top AI researchers are women?
# Data set: names of top AI researchers from Wikipedia
# Guess the gender from first name and estimate the percentage

import random
import nltk
from nltk.corpus import names

def gender_features(name):
    name = name.lower()
    features = {
        'first_letter': name[0],
        'last_letter': name[-1],
        'length': len(name),
        'first_two_letters': name[:2],
        'last_two_letters': name[-2:]
    }
    for letter in 'abcdefghijklmnopqrstuvwxyz':
        features['has_({})'.format(letter)] = letter in name

    return features


def report(clf, train_set, val_set):
    print('Train Set Accuracy:', nltk.classify.accuracy(clf, train_set))
    print('Validation Set Accuracy:', nltk.classify.accuracy(clf, val_set))
    print(clf.show_most_informative_features(5))


male_names = [(name, 'male') for name in names.words('male.txt')]
female_names = [(name, 'female') for name in names.words('female.txt')]
labeled_names = male_names + female_names
random.shuffle(labeled_names)

train_names, val_names, test_names = labeled_names[500:], labeled_names[500:1500], labeled_names[:500]

train_set = [(gender_features(n), gender) for (n, gender) in train_names]
val_set = [(gender_features(n), gender) for (n, gender) in val_names]
test_set = [(gender_features(n), gender) for (n, gender) in test_names]


# Naive Bayes + prefixes/suffixes
clf = nltk.NaiveBayesClassifier.train(train_set)
report(clf, train_set, val_set)
# Train Set Accuracy: 0.816
# Validation Set Accuracy: 0.826

# Estimate the percentage of women among top AI researchers
# Data from https://en.wikipedia.org/wiki/Category:Artificial_intelligence_researchers

with open('ai_researchers.txt') as f:
    data = f.read()

num_women = 0
total = 0
for name in data:
    first_name = data.split(' ')[0]
    total += 1
    if clf.classify(gender_features(name)) == 'female':
        num_women += 1

print('Estimated percentage of women among top AI researchers according to Wikidata: {}'.format(num_women / total * 100))
# 13.6%

