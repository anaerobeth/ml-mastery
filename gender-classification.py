# Identify gender from first names
# Adapted from https://www.nltk.org/book/ch06.html

game_of_thrones = {
    'male': [
        'Eddard', 'Robb',' Jon', 'Bran', 'Jaime', 'Joffrey',
        'Tyrion', 'Theon', 'Varys', 'Viserys'
    ],
    'female': [
        'Catelyn', 'Sansa', 'Arya', 'Cersei', 'Daenerys', 'Melisandre',
        'Margaery', 'Brienne', 'Gilly', 'Missandei'
    ]
}

japanese = {
    'male': [
        'Haruto', 'Yuto', 'Sota', 'Yuki', 'Hayato', 'Haruki',
        'Ryusei', 'Koki', 'Sora', 'Sosuke'
    ],
    'female': [
        'Himari', 'Hina', 'Yua', 'Sakura', 'Ichika', 'Akari',
        'Sara', 'Saeko', 'Aiko', 'Atsuko'
    ]
}

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
    print(clf.show_most_informative_features(3))

    for index, names in enumerate([game_of_thrones, japanese], start=1):
        true_males = 0
        for name in names['male']:
            if clf.classify(gender_features(name)) == 'male':
                true_males += 1

        true_females = 0
        for name in names['female']:
            if clf.classify(gender_features(name)) == 'female':
                true_females += 1

        print('Gender classification of Set {}:'.format(index))
        print('Males: {} out of 10 were correctly classified'.format(true_males))
        print('Females: {} out of 10 were correctly classified'.format(true_females))


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
# Gender classification of Game of Thrones characters:
# Males: 8 out of 10 were correctly classified
# Females: 8 out of 10 were correctly classified

# 2. Naive Bayes + last letter + length and extra features
# Gender classification of Game of Thrones characters:
# Males: 9 out of 10 were correctly classified
# Females: 9 out of 10 were correctly classified

# 3. Naive Bayes#2 + prefixes/suffixes
clf = nltk.NaiveBayesClassifier.train(train_set)
report(clf, train_set, val_set)

""" Results
Train Set Accuracy: 0.817
Validation Set Accuracy: 0.816
Most Informative Features
        last_two_letters = 'na'           female : male   =    101.5 : 1.0
        last_two_letters = 'la'           female : male   =     77.0 : 1.0
        last_two_letters = 'ia'           female : male   =     39.2 : 1.0
Gender classification of Set 1:
Males: 7 out of 10 were correctly classified
Females: 10 out of 10 were correctly classified
Gender classification of Set 2:
Males: 5 out of 10 were correctly classified
Females: 8 out of 10 were correctly classified

Observations: The classifier did well with names of female characters of Game of Thrones but did poorly with male Japanese names.
"""
