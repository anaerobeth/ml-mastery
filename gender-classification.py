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
print('Test Accuracy:', nltk.classify.accuracy(clf, test_set))
# 0.79
print(clf.show_most_informative_features(3))
# Most Informative Features
#            last_letter = 'a'            female : male   =     36.8 : 1.0
#            last_letter = 'k'              male : female =     31.3 : 1.0
#            last_letter = 'f'              male : female =     26.6 : 1.0

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

