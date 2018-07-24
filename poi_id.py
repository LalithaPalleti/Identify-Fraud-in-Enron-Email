#!/usr/bin/python

import sys
import pickle
sys.path.append("../tools/")

from feature_format import featureFormat, targetFeatureSplit
from tester import dump_classifier_and_data

### Task 1: Select what features you'll use.
### features_list is a list of strings, each of which is a feature name.
### The first feature must be "poi".
features_list = ['poi','salary'] # You will need to use more features

### Load the dictionary containing the dataset
with open("final_project_dataset.pkl", "r") as data_file:
    data_dict = pickle.load(data_file)

### Task 2: Remove outliers
data_dict.pop('TOTAL',0)
### Task 3: Create new feature(s)

for key,value in data_dict.items():
    if value['from_poi_to_this_person'] != 'Nan' and value['from_this_person_to_poi'] != 'NaN' and value['to_messages'] != 'NaN' and value['from_messages'] != 'NaN' and value['poi'] != 'NaN':
        
        value['percent_of_to_emails_from_poi'] = (value['from_poi_to_this_person']*100.0)/value['to_messages']
               
        value['percent_of_from_emails_to_poi'] = (value['from_this_person_to_poi']*100.0)/value['from_messages']
               

features_list.append('total_stock_value')
features_list.append('bonus')
features_list.append('from_poi_to_this_person')
features_list.append('from_this_person_to_poi')
features_list.append('deferred_income')
features_list.append('exercised_stock_options')
features_list.append('long_term_incentive')


### Store to my_dataset for easy export below.
my_dataset = data_dict

### Feature Scaling
from sklearn import preprocessing
scaler = preprocessing.MinMaxScaler()
import numpy as np

def feature_scaling(feature):
    temp =[]
    
    for key,value in my_dataset.items():
        if value[feature] != 'NaN':
            temp.append([float(value[feature])])
        elif value[feature] == 'NaN':
            temp.append([0.0])
    
    temp = np.array(temp)
    
    scaler.fit(temp)
    scaled_values = scaler.transform(temp)
    
    index = 0
    for key,value in my_dataset.items():
        value[feature] = (scaled_values[index])[0]
        index += 1

feature_scaling('salary')
feature_scaling('bonus')
feature_scaling('total_stock_value')
feature_scaling('from_poi_to_this_person')
feature_scaling('from_this_person_to_poi')
feature_scaling('deferred_income')
feature_scaling('exercised_stock_options')
feature_scaling('long_term_incentive')

### Extract features and labels from dataset for local testing
data = featureFormat(my_dataset, features_list, sort_keys = True)
labels, features = targetFeatureSplit(data)

### Task 4: Try a varity of classifiers
### Please name your classifier clf for easy export below.
### Note that if you want to do PCA or other multi-stage operations,
### you'll need to use Pipelines. For more info:
### http://scikit-learn.org/stable/modules/pipeline.html

# Provided to give you a starting point. Try a variety of classifiers.
from sklearn.naive_bayes import GaussianNB
clf = GaussianNB()

### Task 5: Tune your classifier to achieve better than .3 precision and recall 
### using our testing script. Check the tester.py script in the final project
### folder for details on the evaluation method, especially the test_classifier
### function. Because of the small size of the dataset, the script uses
### stratified shuffle split cross validation. For more info: 
### http://scikit-learn.org/stable/modules/generated/sklearn.cross_validation.StratifiedShuffleSplit.html

    

# Example starting point. Try investigating other evaluation techniques!
from sklearn.cross_validation import train_test_split
features_train, features_test, labels_train, labels_test = \
    train_test_split(features, labels, test_size=0.3, random_state=42)
    


#GuassianNB
from sklearn.naive_bayes import GaussianNB
from sklearn.grid_search import GridSearchCV
from sklearn.decomposition import RandomizedPCA
from sklearn.pipeline import Pipeline
clf = GaussianNB()
estimators = [('reduce_dim', RandomizedPCA(n_components=4, whiten=True)),('Guassian',clf)]
pipe = Pipeline(estimators)
pipe.fit(features_train,labels_train)
labels_pred = pipe.predict(features_test)

### Task 6: Dump your classifier, dataset, and features_list so anyone can
### check your results. You do not need to change anything below, but make sure
### that the version of poi_id.py that you submit can be run on its own and
### generates the necessary .pkl files for validating your results.

dump_classifier_and_data(clf, my_dataset, features_list)