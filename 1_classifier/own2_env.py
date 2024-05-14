from sklearn import svm
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
import nltk
import joblib
from Util import OneHot
from nltk import word_tokenize, pos_tag, ne_chunk, tree
import pandas as pd
from sklearn.feature_extraction import DictVectorizer
from sklearn.model_selection import train_test_split
import spacy
import keras
from keras.utils import plot_model
from keras.models import Model
from keras.layers import Input, SpatialDropout1D, Conv1D, Flatten, Dense, Activation, Add, MaxPooling1D
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt
import pickle
import re
# Load the spaCy language model
nlp = spacy.load('en_core_web_sm')

def extract_features(sentence):
    # Parse the sentence using spaCy
    doc = nlp(sentence)
    flag = False
    # Extract syntactic features
    flag_open = '<details>'
    flag_close = '</details>'
    features = {}
    threshold = 5
    '''
    features['num_punctuations'] = len([token for token in doc if token.is_punct])
    features['num_words'] = len(doc)
    features['num_chars'] = sum(len(token) for token in doc if not token.is_space)
    features['avg_word_length'] = features['num_chars'] / features['num_words'] if features['num_words'] > 0 else 0
    features['lexical_diversity'] = len(set(token.text for token in doc)) / features['num_words'] if features['num_words'] > 0 else 0
    '''
    if flag_open in sentence:
        flag = True
    if flag_close in sentence:
        flag = False
    if ': ' in doc.text and any(char.isdigit() for char in doc.text) and flag==False:
        features['has_colon'] = 1
    else:
        features['has_colon'] = 0
    
    #features['contains_digit'] = any(char.isdigit() for char in sentence)
    # Check if the sentence starts with "OS version:"
    #features['num_chars'] = sum(len(token) for token in doc if not token.is_space)
    features['num_words'] = len(doc)
    #features['num_verbs'] = len([token for token in doc if token.pos_ == 'VERB'])
    #features['avg_word_length'] = features['num_chars'] / features['num_words'] if features['num_words'] > 0 else 0

    if doc.text.startswith('OS version' or 'OS'):
        features['is_os_version'] = 1
    else:
        features['is_os_version'] = 0
    # Check if the sentence starts with "OS version:"
    if (doc.text.startswith('VSCode' or 'VS Code version' or 'VSCode version' or "V8") and features['num_words'] <= threshold):
        features['is_vs_version'] = 1
    else:
        features['is_vs_version'] = 0
    keywords = ["version", 'insider', 'os', 'software', 'generic', 'Linux', "Windows", "OS", 'x64']
    for keyword in keywords:
        if keyword in doc.text and any(char.isdigit() for char in doc.text) and flag == False:
            features["has_keyword"] = 1
        if keyword in doc.text and features['num_words'] > threshold:
            features["has_keyword"] = 0
    if '|' in sentence and flag == True or (doc.text.startswith("Steps" or "flash" or "web")):
        features['palka'] = 0
    else:
        features['palka'] = 1
    for ent in doc.ents:
        if ent.label_ in ['CARDINAL', 'PRODUCT']:
            features['has_version_number'] = 1
            break
        if ent.label_ in ['CARDINAL', 'PRODUCT'] and features['num_words'] > threshold:
            features['has_version_number'] = 0
    
    if re.match(r"^\d+[.)]", sentence):
        features["numerated"] = 0
    else:
        features["numerated"] = 1
    
    question = sentence.endswith("?")
    if question == True:
        features["question"] = 0
    else: 
        features["question"] = 1
    
    '''
    features = {
        'num_nouns': len([token for token in doc if token.pos_ == 'NOUN']),
        'num_verbs': len([token for token in doc if token.pos_ == 'VERB']),
        'num_adjectives': len([token for token in doc if token.pos_ == 'ADJ']),
        'num_adverbs': len([token for token in doc if token.pos_ == 'ADV']),
        'num_pronouns': len([token for token in doc if token.pos_ == 'PRON']),
        'num_prepositions': len([token for token in doc if token.pos_ == 'ADP']),
        'num_conjunctions': len([token for token in doc if token.pos_ == 'CONJ']),
        'num_digits': len([token for token in doc if token.is_digit]),
        'num_punctuations': len([token for token in doc if token.is_punct]),
        'num_stopwords': len([token for token in doc if token.is_stop])
    }
    '''
    return features

def block(pre, num_filters):
    x = Activation(activation='relu')(pre)
    x = Conv1D(filters=num_filters, kernel_size=3, padding='same', strides=1)(x)
    x = Activation(activation='relu')(x)
    x = Conv1D(filters=num_filters, kernel_size=3, padding='same', strides=1)(x)
    x = Add()([x, pre])
    return x

def SaveVar(train_feature, test_feature, train, test):
    joblib.dump(train_feature, "train_feature.pkg")
    joblib.dump(test_feature, "test_feature.pkg")
    joblib.dump(train, "train.pkg")
    joblib.dump(test, "test.pkg")

def LoadDPCNNRFC():
    model = keras.models.load_model("Bert_DPCNN_sam_sec.h5")  # Load Bert+DPCNN
    layermodel = Model(inputs=model.input, outputs=model.layers[-2].output)
    #rfc_model = joblib.load("rfc.pth")
    return layermodel

def LoadVar():
    train_feature = joblib.load("train_feature.pkg")
    test_feature = joblib.load("test_feature.pkg")
    train = joblib.load("train.pkg")
    test = joblib.load("test.pkg")
    return train_feature, test_feature, train, test

def DPCNN_Compressed_Vec_Gen(feature, layermodel):
    compressedFeature = layermodel.predict(feature)
    return compressedFeature

def rfc_Train(trainvec, trainlabel):
    rfc_model = RandomForestClassifier(min_samples_split=10, n_estimators = 100, max_features="auto")
    reOneHotTrainLabel = reOneHot(trainlabel)
    rfc_model.fit(trainvec, reOneHotTrainLabel)
    print("yogi")
    joblib.dump(rfc_model, "rfc.pth")

def rfc_Test(testvec, testlabel):
    rfc_model = joblib.load("rfc.pth")
    reOneHotTestLabel = reOneHot(testlabel)
    predictLabel = rfc_model.predict(testvec)
    Accuracy = accuracy_score(reOneHotTestLabel, predictLabel)
    print("RFC_Accuracy:\n", Accuracy)
    return Accuracy

def reOneHot(label):
    imptLabelList = []
    for i in label:
        if i[0] == 1:
            imptLabelList.append(0)
        else:
            imptLabelList.append(1)
    reOneHotLabel = np.array(imptLabelList)
    return reOneHotLabel

# Load your data into a pandas DataFrame, where the 'sentences' column contains the sentences
# and the 'label' column contains the binary labels (0 or 1)
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
import joblib
data = pd.read_csv("SO-ModelTrainData-env_extended.csv", encoding= 'unicode_escape')

# Apply the extract_features function to each sentence and store the resulting feature dictionary in a list
features_list = []
for sentence in data['Sentence']:
    features = extract_features(sentence)
    features_list.append(features)

# Convert the list of feature dictionaries into a feature matrix using DictVectorizer
vectorizer = DictVectorizer()
X = vectorizer.fit_transform(features_list)
#X = X.reshape((X.shape[0], 1, X.shape[1]))
#X = np.reshape(X.toarray(), (len(features_list), 1, -1)) #X = np.reshape(X.toarray(), (X.shape[0], 1, X.shape[1]))
X = np.reshape(X, (X.shape[0], -1))
print("SHAPES", X.shape)
#exit()
# Convert the binary labels to a numpy array
y = data['Opinion'].values
y = np.array(y)
# Split the data into training and testing sets

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
'''
# Define the parameter grid to search over
param_grid = {
    'n_estimators': [50, 100, 150],
    'max_depth': [None, 5, 10],
    'min_samples_split': [2, 5],
    'min_samples_leaf': [1, 2]
}

# Create a RandomForestClassifier object
rf = RandomForestClassifier(random_state=42)

# Perform grid search with cross-validation
grid_search = GridSearchCV(rf, param_grid=param_grid, cv=5)


grid_search.fit(X_train, y_train)

# Get the best hyperparameters
best_params = grid_search.best_params_

# Train the model on the best hyperparameters
rf_best = RandomForestClassifier(**best_params, random_state=42)


rf_best.fit(X_train, y_train)
'''
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)
joblib.dump(vectorizer, "vectorizer_ext_7feat_new.pkl")

# Save the trained model
with open('random_forest_7feat_new.pkl', 'wb') as file:
    pickle.dump(rf, file)


# Predict on the test set
#y_pred = rf_best.predict(X_test)

print("done")
