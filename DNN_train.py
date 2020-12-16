import os 
import csv
import tensorflow as tf 
from keras.models import Sequential
from keras.layers import Dense, Activation
import numpy as np 
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics
from keras.utils import np_utils  
import time 

# tensorflow env setting 
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
gpu_options = tf.GPUOptions(allow_growth=True)
sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
tf.keras.backend.set_session(sess)

# DNN parameter setting
batch_size = 2000
epochnum = 40
# The feature you want to test. 
feature_list = [0, 0, 1, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0]
feature_num = feature_list.count(1)

# Build DNN model. 
def Build_Deep_Neural_Network(batch_size,epochnum):
  fp = open("sol.txt", "r")
  lines = fp.read()
  sol = lines.split(' ')
  print(sol)
  model = Sequential()
  model.add(Dense(units=feature_num, input_dim=feature_num, kernel_initializer='normal', activation='relu'))
  hiddenlayer = int(sol[0])
  unitslist = []
  for i in range(hiddenlayer) :
    unitslist.append(int(float(sol[i+1])))
  for j in range( int(sol[0]) ) :
    model.add(Dense(units = unitslist[j], kernel_initializer='normal', activation = 'relu'))
  model.add(Dense(units=2, kernel_initializer='normal', activation='softmax'))
  model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy']) 
  return model 

# Get the train and test data.
def Get_right_feature(feature_list) :
    with open( "2class_NSL-KDDTrain+.txt") as txtfile:
        rows = csv.reader(txtfile)
        train_data = []
        train_label = []
        for row in rows:
            temp = []
            for i in range(len(feature_list)) :
                if feature_list[i] == 1 :
                    temp.append(row[i])    
            train_data.append(temp)
            train_label.append(row[41])
    with open( "2class_NSL-KDDTest+.txt") as txtfile:
        rows = csv.reader(txtfile)
        test_data = []
        test_label = []
        for row in rows:
            temp = []
            for i in range(len(feature_list)) :
                if feature_list[i] == 1 :
                    temp.append(row[i])    
            test_data.append(temp)
            test_label.append(row[41])
    with open( "2class_NSL-KDDTest-21+.txt") as txtfile:
        rows = csv.reader(txtfile)
        test21_data = []
        test21_label = []
        for row in rows:
            temp = []
            for i in range(len(feature_list)) :
                if feature_list[i] == 1 :
                    temp.append(row[i])    
            test21_data.append(temp)
            test21_label.append(row[41])
    return train_data, train_label, test_data, test_label, test21_data, test21_label

# Caculate the score.
def metrice(predict_result, answer):
    TP = 0 
    TN = 0 
    FP = 0 
    FN = 0 
    for i in range(len(answer)) :

        if(answer[i]=='0' and ( predict_result[i] == 0 or predict_result[i] == '0') ) :
            TN +=1 
        elif (answer[i]=='1' and ( predict_result[i] == 0 or predict_result[i] == '0') ) :

            FN +=1
        elif (answer[i]=='1' and ( predict_result[i] == 1 or predict_result[i] == '1')) :

            TP +=1
        elif (answer[i]=='0' and ( predict_result[i] == 1 or predict_result[i] == '1')) :

            FP +=1
    Accuracy = ( TP + TN ) / (TP + TN + FP + FN) 
    Recall = TP / (TP + FN)
    Precision = TP / ( TP + FP )
    score = []
    score.append(Accuracy)
    score.append(Recall)  
    score.append(Precision)
    score.append(TP)
    score.append(TN)
    score.append(FP)
    score.append(FN)
    return score

# Update record.
def DNN_result_update(normal_score,diff_score) :
    fp=open("normal_accuracy_DNN.txt","a")
    fp.write(str(normal_score[0]) + "\n")
    fp.close()
    fp=open("normal_recall_DNN.txt","a")
    fp.write(str(normal_score[1]) + "\n")
    fp.close()
    fp=open("normal_precision_DNN.txt","a")
    fp.write(str(normal_score[2]) + "\n")
    fp.close()
    fp=open("diff_accuracy_DNN.txt","a")
    fp.write(str(diff_score[0]) + "\n")
    fp.close()
    fp=open("diff_recall_DNN.txt","a")
    fp.write(str(diff_score[1]) + "\n")
    fp.close()
    fp=open("diff_precision_DNN.txt","a")
    fp.write(str(diff_score[2]) + "\n")
    fp.close()
    fp=open("normal_metrice_DNN.txt","a")
    fp.write(str(normal_score[3])+", "+str(normal_score[4])+", "+str(normal_score[5])+", "+str(normal_score[6])+"\n")
    fp.close()
    fp=open("diff_metrice_DNN.txt","a")
    fp.write(str(diff_score[3]) + ", "+str(diff_score[4]) + ", "+str(diff_score[5]) + ", "+str(diff_score[6]) + "\n")
    fp.close()

# mian :
train_data, train_label, test_data, test_label, test21_data, test21_label = Get_right_feature(feature_list)
print(train_data[0])
start_time = time.time()
DNN_model = Build_Deep_Neural_Network(batch_size, epochnum)
x_ = np.array(train_data)
y_ = np.array(train_label)
trainlabel_OneHot = np_utils.to_categorical(y_)
DNN_model.fit(x_, trainlabel_OneHot,epochs=epochnum , batch_size=batch_size, verbose=1)
result_1 = DNN_model.predict_classes(np.array(test_data))
result_2 = DNN_model.predict_classes(np.array(test21_data))
normal_score = []
diff_score = []
normal_score = metrice(result_1,test_label)
diff_score = metrice(result_2,test21_label)
DNN_result_update(normal_score, diff_score)