# Identify gender from first names
# Adapted from https://www.nltk.org/book/ch06.html

import random
import nltk
from nltk.corpus import names

def gender_features(name):
    name = name.lower()
    features = {
        'first_letter': name[0],
        'last_letter': name[-1],
        'length': len(name)
    }
    for letter in 'abcdefghijklmnopqrstuvwxyz':
        features['has_({})'.format(letter)] = letter in name

    return features


def inspect_errors(val_names):
    errors = []
    for (name, tag) in val_names:
        guess = clf.classify(gender_features(name))
        if guess != tag:
            errors.append( (tag, guess, name) )
    return errors

male_names = [(name, 'male') for name in names.words('male.txt')]
female_names = [(name, 'female') for name in names.words('female.txt')]
labeled_names = male_names + female_names
random.shuffle(labeled_names)

train_names, val_names, test_names = labeled_names[500:], labeled_names[500:1500], labeled_names[:500]

train_set = [(gender_features(n), gender) for (n, gender) in train_names]
val_set = [(gender_features(n), gender) for (n, gender) in val_names]
test_set = [(gender_features(n), gender) for (n, gender) in test_names]


# 1. Naive Bayes + last letter
# Test Set Accuracy: 0.79
# Most Informative Features
#            last_letter = 'a'            female : male   =     36.8 : 1.0
#            last_letter = 'k'              male : female =     31.3 : 1.0
#            last_letter = 'f'              male : female =     26.6 : 1.0
#
# Gender classification of Game of Thrones characters:
# Males: 8 out of 10 were correctly classified
# Females: 8 out of 10 were correctly classified

# 2. Naive Bayes + last letter + length and extra features
clf = nltk.NaiveBayesClassifier.train(train_set)
print('Train Set Accuracy:', nltk.classify.accuracy(clf, train_set))
Train Set Accuracy: 0.787
print('Test Set Accuracy:', nltk.classify.accuracy(clf, test_set))
Test Set Accuracy: 0.786

print(clf.show_most_informative_features(3))
# Most Informative Features
#            last_letter = 'k'              male : female =     44.0 : 1.0
#            last_letter = 'a'            female : male   =     35.4 : 1.0
#            last_letter = 'f'              male : female =     16.0 : 1.0

male_characters = ['Eddard', 'Robb',' Jon', 'Bran', 'Jaime', 'Joffrey', 'Tyrion', 'Theon', 'Varys', 'Viserys']
female_characters = ['Catelyn', 'Sansa', 'Arya', 'Cersei', 'Daenerys', 'Melisandre', 'Margaery', 'Brienne', 'Gilly', 'Missandei']

true_males = 0
for name in male_characters:
    if clf.classify(gender_features(name)) == 'male':
        true_males += 1

true_females = 0
for name in female_characters:
    if clf.classify(gender_features(name)) == 'female':
        true_females += 1

print('Gender classification of Game of Thrones characters:')
print('Males: {} out of 10 were correctly classified'.format(true_males))
print('Females: {} out of 10 were correctly classified'.format(true_females))
# Gender classification of Game of Thrones characters:
# Males: 9 out of 10 were correctly classified
# Females: 9 out of 10 were correctly classified
