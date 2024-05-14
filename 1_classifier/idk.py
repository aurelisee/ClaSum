from idk_util import readcsv, SheetName, stopwordinput, preprocess_spacy, wirteDataList, EmptyCheck, DataListListProcess, DataListList2float
import re
from bert_serving.client import BertClient
from idk_ import SOPredict, LoadDPCNNRFC, TextRank, TitleSimilarity
import math
from idk_util import DataListList2float, DataListList2int, StrList2FloatList, StrList2IntList, DataListListProcess, wirteDataList
import pandas as pd
import spacy
from idk_believe import *
from sklearn.metrics import accuracy_score, recall_score, f1_score, precision_score, precision_recall_fscore_support
from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_extraction.text import CountVectorizer
import pickle
import joblib

def DataInput(datasetName):
    filepath = "BugSum_Data_"+str(datasetName)+".xlsx"
    sheetNameList = SheetName(filepath)

    dataList = []

    io = pd.io.excel.ExcelFile(filepath, engine='openpyxl')

    for sigSheet in sheetNameList:
        #print("sigSheet", sigSheet)
        data = readcsv(filepath, sigSheet, io)

        dataList.append(data.copy())

    return sheetNameList, dataList

# C:/Users/user/Desktop/shared_bs/Other_FIles/SOModelData/SavedModels/rfc.pth

def DataOPscoreEnv(dataList):
    print("DataOPscore: ")
    sheetNum = len(dataList)
    #dpcnn, rfc = LoadDPCNNRFC()
    nlp = spacy.load("en_core_web_sm")
    #vectorizer = DictVectorizer()
    loaded_model = joblib.load('random_forest_7feat.pkl')
    # Load the feature vectorizer
    vectorizer = joblib.load('vectorizer_ext_7feat.pkl')
    #dpcnn = LoadDPCNNRFC()
    #with open("random_forest.pkl", "rb") as f:
    #    loaded_model = pickle.load(f)
    sheetNumber = len(dataList)
    new_data_features = []
    numberBR = 0
    averageAcc = 0
    averageRecall = 0
    averageFscore = 0
    for i in range(sheetNum):
        #print("OPscore Processing:", i, "/", sheetNum)
        #senVecList = DataListList2float(dataList[i]["SenVec"])
        #print(senVecList)
        #exit()
        numberBR += 1
        SenList = dataList[i]["Sentence"]
        new_data_features = [extract_features(str(sentence) for sentence in SenList]

        # Vectorize the features
        X_new = vectorizer.transform(new_data_features)

        # Get the feature names
        feature_names = vectorizer.get_feature_names()

        # Convert the sparse matrix to a pandas dataframe
        X_new_df = pd.DataFrame.sparse.from_spmatrix(X_new, columns=feature_names)

        # Make predictions on the new data
        predictions = loaded_model.predict(X_new_df)

        
        goldenList = StrList2IntList(dataList[i]["GoldenEnv"])
        #print(predictions.tolist(),"\ngoldenlist",goldenList)
        #accuracy = precision_score(goldenList, predictions.tolist())
        #recall = recall_score(goldenList, predictions.tolist(), average='weighted')
        #fscore = f1_score(goldenList, predictions.tolist(), average='weighted')
        res = precision_recall_fscore_support(goldenList, predictions.tolist(), average='binary',zero_division=1)
        averageAcc = averageAcc + res[0]
        averageRecall = averageRecall + res[1]
        averageFscore = averageFscore + res[2]
        #acc, recall, f_score = AccuracyMeasure(predictions, goldenList)
        dataList[i]["Environment"] = predictions
        dataList[i]["Performance"] = res#.copy()
    
    averageAcc = averageAcc/numberBR
    averageRecall = averageRecall/numberBR
    #averageFscore = 2 * averageAcc * averageRecall / (averageAcc + averageRecall)
    averageFscore = averageFscore/numberBR
    print("Precision", averageAcc)
    print("Recall", averageRecall)
    print("Fscore", averageFscore)
    return dataList


def DataOPscoreReproSteps(dataList):
    print("DataOPscore: ")
    sheetNum = len(dataList)
    #dpcnn, rfc = LoadDPCNNRFC()
    nlp = spacy.load("en_core_web_sm")
    #vectorizer = DictVectorizer()
    loaded_model = joblib.load('random_forest_repro_steps.pkl')
    # Load the feature vectorizer
    vectorizer = joblib.load('random_forest_vectorizer_repro_steps.pkl')
    #dpcnn = LoadDPCNNRFC()
    #with open("random_forest.pkl", "rb") as f:
    #    loaded_model = pickle.load(f)
    sheetNumber = len(dataList)
    new_data_features = []
    numberBR = 0
    averageAcc = 0
    averageRecall = 0
    averageFscore = 0
    for i in range(sheetNum):
        #print("OPscore Processing:", i, "/", sheetNum)
        #senVecList = DataListList2float(dataList[i]["SenVec"])
        #print(senVecList)
        #exit()
        numberBR += 1
        SenList = dataList[i]["Sentence"]
        new_data_features = [extract_features(str(sentence)) for sentence in SenList]

        # Vectorize the features
        X_new = vectorizer.transform(new_data_features)

        # Get the feature names
        feature_names = vectorizer.get_feature_names()

        # Convert the sparse matrix to a pandas dataframe
        X_new_df = pd.DataFrame.sparse.from_spmatrix(X_new, columns=feature_names)

        # Make predictions on the new data
        predictions = loaded_model.predict(X_new_df)
        dataList[i]["ReproductionStep"] = predictions
        
        goldenList = StrList2IntList(dataList[i]["GoldenRepro"])
        #print(predictions.tolist(),"\ngoldenlist",goldenList)
        #accuracy = precision_score(goldenList, predictions.tolist())
        #recall = recall_score(goldenList, predictions.tolist(), average='weighted')
        #fscore = f1_score(goldenList, predictions.tolist(), average='weighted')
        res = precision_recall_fscore_support(goldenList, predictions.tolist(), average='binary',zero_division=1)
        averageAcc = averageAcc + res[0]
        averageRecall = averageRecall + res[1]
        averageFscore = averageFscore + res[2]
        #acc, recall, f_score = AccuracyMeasure(predictions, goldenList)
        
        dataList[i]["Performance_Repro"] = res#.copy()
    
    averageAcc = averageAcc/numberBR
    averageRecall = averageRecall/numberBR
    #averageFscore = 2 * averageAcc * averageRecall / (averageAcc + averageRecall)
    averageFscore = averageFscore/numberBR
    print("Precision", averageAcc)
    print("Recall", averageRecall)
    print("Fscore", averageFscore)
    return dataList
    
nlp = spacy.load('en_core_web_sm')

def extract_features_repro(sentence):
    # Parse the sentence using spaCy
    features = {}
    doc = nlp(sentence)
    
    for token in doc:
        if token.pos_ == "VERB":
            for child in token.children:
                if child.dep_ == "dobj":
                    features["verb_obj"] = 1
                else:
                    features["verb_obj"] = 0
    # If no match was found, return None
    # Extract syntactic features
    
    for token in doc:
        if token.pos_ == "VERB" and token.dep_ == "ROOT" and token.tag_ == "VB":
            features["imperative"] = 1
        else:
            features["imperative"] = 0
    
    verb_pattern = re.compile(r"VB|VBD|VBG|VBN")
    # Check if the sentence contains a verb that is a root of a clause
    for token in doc:
        if verb_pattern.match(token.tag_) and token.dep_ == "ROOT":
            features["verbform"] = 1
        else:
            features["verbform"] = 0
    
    if re.match(r"^\d+[.)]", sentence):
        features["numerated"] = 1
    else:
        features["numerated"] = 0

    question = sentence.endswith("?")
    if question == True:
        features["question"] = 0
        features["numerated"] = 0
        features["verbform"] = 0
        features["imperative"] = 0
        features["verb_obj"] = 0
    '''
    features['num_punctuations'] = len([token for token in doc if token.is_punct])
    features['num_words'] = len(doc)
    features['num_chars'] = sum(len(token) for token in doc if not token.is_space)
    features['avg_word_length'] = features['num_chars'] / features['num_words'] if features['num_words'] > 0 else 0
    features['lexical_diversity'] = len(set(token.text for token in doc)) / features['num_words'] if features['num_words'] > 0 else 0
    
    if ': ' in doc.text:
        features['has_colon'] = 1
    else:
        features['has_colon'] = 0
    '''
    # Check if the sentence starts with "OS version:"
    #features['num_chars'] = sum(len(token) for token in doc if not token.is_space)
    features['num_words'] = len(doc)
    #features['avg_word_length'] = features['num_chars'] / features['num_words'] if features['num_words'] > 0 else 0
    '''
    if doc.text.startswith('OS version:' or 'OS:'):
        features['is_os_version'] = 1
    else:
        features['is_os_version'] = 0
    # Check if the sentence starts with "OS version:"
    if doc.text.startswith('VSCode:' or 'VS Code version:' or 'VSCode version:'):
        features['is_vs_version'] = 1
    else:
        features['is_vs_version'] = 0
    keywords = ["version", 'insider', 'os', 'software', 'generic', 'Linux', "Windows", "OS", 'x64']
    for keyword in keywords:
        if keyword in sentence:
            features["has_keyword"] = 1
        else:
            features["has_keyword"] = 0
    for ent in doc.ents:
        if ent.label_ in ['CARDINAL', 'PRODUCT']:
            features['has_version_number'] = 1
            break
        else:
            features['has_version_number'] = 0
    '''
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
    if len(features) != 6:
        features['bullshit'] = 0
    return features


def extract_features_env(sentence):
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
    keywords = ["version", 'insider', 'os', 'software', 'generic', 'Linux', "Windows", "OS", 'x64', 'VSCode']
    for keyword in keywords:
        if keyword in doc.text and any(char.isdigit() for char in doc.text) and flag == False:
            features["has_keyword"] = 1
        if keyword in doc.text and features['num_words'] > threshold:
            features["has_keyword"] = 0
    if '|' in sentence and flag == True: #or (doc.text.startswith("Steps" or "flash" or "web")):
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
    
    if "enabled" in sentence or "|" in sentence:
        features['has_version_number'] = 0
        features["numerated"] = 0
        features["has_keyword"] = 0
        features['is_vs_version'] = 0
        features['is_os_version'] = 0
        features['has_colon'] = 0
    if len(sentence) > 45:
        features['has_version_number'] = 0
        features["numerated"] = 0
        features["has_keyword"] = 0
        features['is_vs_version'] = 0
        features['is_os_version'] = 0
        features['has_colon'] = 0
    if '"' in sentence:
        features['has_version_number'] = 0
        features["numerated"] = 0
        features["has_keyword"] = 0
        features['is_vs_version'] = 0
        features['is_os_version'] = 0
        features['has_colon'] = 0
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
    if len(features) != 9:
        features['extra'] = 0
    return features

def AccuracyMeasure(selectedList, goldenList):
    tCounter = 0
    senLen = len(goldenList)
    # wordRes 크기랑 안맞어서 내가 +1 추가함
    selectedSen = len(selectedList)
    print(selectedSen)

    for i in range(senLen):
        if goldenList[i] == 1:
            tCounter = tCounter + 1

    if (tCounter ==0):
        return 1

    counter = 0
    for i in range(senLen):
        if selectedList[i] == 1 and goldenList[i] == 1:
            counter = counter + 1

    recall = counter*1.0/tCounter
    acc = counter*1.0/selectedSen

    # print("golden size", tCounter)
    # print("wordres", wordRes)
    # print("selected sen", selectedSen)

    if recall + acc == 0:
        fscore = 0
    else:
        fscore = 2 * recall * acc / (recall + acc)
    return acc, recall, fscore


def DataPreprocess(datasetName):
    sheetNameList, dataList = DataInput(datasetName)
    print("Start")

    dataList = DataOPscore(dataList) # STep3
    wirteDataList(dataList, sheetNameList, datasetName)
    print("Finish OPscore")
    
    
    wirteDataList(dataList, sheetNameList, datasetName)


if __name__ == "__main__":
    # SDS 데이터셋의 경우
    #DataPreprocess("SDS")
    DataPreprocess("DDS_new_allclassified")
    #EvaluationBehaviorCapture("Relation", 0.08)
    #DataPreprocess("EST")














