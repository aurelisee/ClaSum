import numpy as np
import pandas as pd
from tensorflow.keras.utils import plot_model
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, SpatialDropout1D, Conv1D, Flatten, Dense, Activation, Add, MaxPooling1D
import tensorflow.keras
from sklearn.model_selection import train_test_split
from sklearn.metrics.pairwise import cosine_similarity
# from models.data_process import label2vec, vec2label
# from APSEC_SO.proj.models.data_process import label2vec, vec2label, one_hot
from idk_util import OneHot
import networkx as nx
from sklearn.pipeline import Pipeline
from tensorflow.keras import backend as k
from sklearn.ensemble import RandomForestClassifier
import joblib
from sklearn.model_selection import StratifiedKFold, GridSearchCV, cross_validate, LeaveOneGroupOut
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt
from tensorflow.keras.layers import Embedding
from bert_serving.client import BertClient

def LoadDPCNNRFC():
    model = tensorflow.keras.models.load_model("random_forest_model.pkl")  # Load Bert+DPCNN
    #model = tensorflow.keras.models.load_model("SOModelData/Bert_DPCNN.h5")  # Load Bert+DPCNN
    '''
    layermodel = Model(inputs=model.input, outputs=model.layers[-2].output)
    try:
        rfc_model = joblib.load("rfc.pth")
        #rfc_model = joblib.load("SOModelData/rfc.pth")
    except:
        print("There is no rfc.pth")
        rfc_model = None
    '''
    return model
    #return layermodel, rfc_model

def TextRank(SenVecList):
    SenVecList = np.array(SenVecList)
    sim_mat = []
    sim_mat = similarity_matrix(SenVecList, len(SenVecList[0]))
    scores = calculate_score(sim_mat)
    scores = list(scores.values())
    # 점수 대신 rank score 사용
    ranked_scores = ranked_sentences(scores)
    #print(ranked_scores)
    #print(scores)
    return scores, ranked_scores
    # 점수 사용 (0.01보다 작음)
    #return scores

def TitleSimilarity(SenVecList, titleVec):
    SenVecList = np.array(SenVecList)
    titleVec = np.array(titleVec)
    embedding_dim = len(titleVec[0])
    #print(titleVec)
    topic_score= np.zeros([len(SenVecList)])
    for i in range(len(SenVecList)):
        topic_score[i] = cosine_similarity(SenVecList[i].reshape(1, embedding_dim),titleVec[0].reshape(1, embedding_dim))[0,0]
    #print("toic score", len(topic_score))
    return topic_score

def SOPredict(SenVecList, layermodel):#def SOPredict(SenVecList, layermodel, rfc_model):
    
    SenVecList = np.array(SenVecList)
    #print(SenVecList.shape)
    #print(SenVecList)
    print("Yogi")
    SenVecList = SenVecList[:, np.newaxis, :]

    '''
    Compressed_Feature = DPCNN_Compressed_Vec_Gen(SenVecList, layermodel)

    predictPos = rfc_model.predict_proba(Compressed_Feature)
    '''
    predictions = layermodel.predict(SenVecList)
    print(predictions)
    #return predictions[:, 0]#return predictPos[:, 0]

# 문장 벡터들 간의 코사인 유사도를 구한 유사도 행렬 생성
# 유사도 행렬의 크기는 (문장 개수 × 문장 개수)
def similarity_matrix(sentence_embedding, embedding_dim):
  sim_mat = np.zeros([len(sentence_embedding), len(sentence_embedding)])


  for i in range(len(sentence_embedding)):
      for j in range(len(sentence_embedding)):
        sim_mat[i][j] = cosine_similarity(sentence_embedding[i].reshape(1, embedding_dim),
                                          sentence_embedding[j].reshape(1, embedding_dim))[0,0]
  return sim_mat

def calculate_score(sim_matrix):
    nx_graph = nx.from_numpy_array(sim_matrix)
    scores = nx.pagerank(nx_graph)

    return scores

def ranked_sentences(scores):
    ranked_scores = sorted(scores)

    text_rank = []
    for i in range(len(scores)):
        text_rank.append(ranked_scores.index(scores[i])+1)

    return text_rank

def DPCNN_Compressed_Vec_Gen(feature, layermodel):
    # print("DPCNN_Compressed_Vec_Gen feature/n", feature)

    compressedFeature = layermodel.predict(feature)

    return compressedFeature