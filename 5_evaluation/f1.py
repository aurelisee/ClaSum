import collections
import re
import string
from csv import writer
import numpy as np

model = 'primera'

def _f1_score(target, prediction):
  """Computes token f1 score for a single target and prediction."""
  prediction_tokens = prediction.split()
  target_tokens = target.split()
  common = (collections.Counter(prediction_tokens) &
            collections.Counter(target_tokens))
  num_same = sum(common.values())
  if num_same == 0:
    return 0
  precision = 1.0 * num_same / len(prediction_tokens)
  recall = 1.0 * num_same / len(target_tokens)
  f1 = (2 * precision * recall) / (precision + recall)
  return {'f1':f1, 'precision': precision, 'recall': recall}

for i in range(1,97):  
  try:
    summary = open('C:/Users/user/Downloads/bug_summarization-master/bug_summarization-master/dataset/result/summary_ADS/'+str(i)+".txt", "r", encoding="UTF-8")
    summaryContent = summary.read().strip()

    ref = open('C:\\Users\\user\\Desktop\\SDSADS_modif\\ADS\\golden_ADS\\w_class\\wo_class'+str(i)+".txt", "r", encoding="UTF-8")
    refContent = ref.read().strip()

    #ref_second_anotated = open('C:/Users/User/Documents/bug_summarization/dataset/result/ref/'+str(i)+"/2.txt", "r", encoding="UTF-8")
    #refSecondAnotatedContent = ref_second_anotated.read().strip()

    f1 = _f1_score(refContent, summaryContent)
    #f1SecondAnotated = _f1_score(refSecondAnotatedContent, summaryContent)

    print(f1)
    print(i, f1['precision'], f1['recall'], f1['f1'])
    list_data=[i, f1['precision'], f1['recall'], f1['f1']]

    with open("C:/Users/user/Downloads/bug_summarization-master/bug_summarization-master/dataset/result/summary_ADS/prf_.csv", 'a', newline='') as f_object:  
        writer_object = writer(f_object)
        writer_object.writerow(list_data)  
        f_object.close()
  except:
    print('skip')
