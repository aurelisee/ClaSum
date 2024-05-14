import subprocess
from csv import writer
for i in range(1,97):    
    r1 = subprocess.getstatusoutput(f"java -jar C:/Users/user/Downloads/C_ROUGE.jar C:/Users/user/Downloads/bug_summarization-master/bug_summarization-master/dataset/result/summary_ADS/"+str(i)+".txt C:\\Users\\user\\Desktop\\SDSADS_modif\\ADS\\ref_ADS_w\\"+str(i)+" 1 A R")
    r2 = subprocess.getstatusoutput(f"java -jar C:/Users/user/Downloads/C_ROUGE.jar C:/Users/user/Downloads/bug_summarization-master/bug_summarization-master/dataset/result/summary_ADS/"+str(i)+".txt C:\\Users\\user\\Desktop\\SDSADS_modif\\ADS\\ref_ADS_w\\"+str(i)+" 2 A R")
    print([i, r1, r2])
    f = {'rouge1':r1, 'rouge2': r2}
    list_data=[i, f['rouge1'], f['rouge2']]
    
    with open("C:/Users/user/Downloads/bug_summarization-master/bug_summarization-master/dataset/result/summary_ADS/Rouge.csv", 'a', newline='') as f_object:  
        writer_object = writer(f_object)
        writer_object.writerow(list_data)  
        f_object.close()
    