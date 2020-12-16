import csv 

# set the time of result you run.
times = 100 


with open( "normal_accuracy_DNN.txt") as txtfile:
    rows = csv.reader(txtfile)
    normal_accuracy = 0 
    for row in rows:
        normal_accuracy = normal_accuracy + float(row[0])    
    normal_accuracy = normal_accuracy / times


with open( "normal_precision_DNN.txt") as txtfile:
    rows = csv.reader(txtfile)
    normal_precision = 0 
    for row in rows:
        normal_precision = normal_precision + float(row[0])   
    normal_precision = normal_precision / times

with open( "normal_recall_DNN.txt") as txtfile:
    rows = csv.reader(txtfile)
    normal_recall = 0 
    for row in rows:
        normal_recall = normal_recall + float(row[0])  
    normal_recall = normal_recall / times


with open( "normal_metrice_DNN.txt") as txtfile:
    rows = csv.reader(txtfile)
    normal_TP = 0 
    normal_TN = 0 
    normal_FP = 0 
    normal_FN = 0 
    for row in rows:
        normal_TP = normal_TP + float(row[0])  
        normal_TN = normal_TN + float(row[1])  
        normal_FP = normal_FP + float(row[2])  
        normal_FN = normal_FN + float(row[3])  
    normal_TP = normal_TP /times  
    normal_TN = normal_TN /times  
    normal_FP = normal_FP/times 
    normal_FN = normal_FN /times  


with open( "diff_accuracy_DNN.txt") as txtfile:
    rows = csv.reader(txtfile)
    diff_accuracy = 0 
    for row in rows:
        diff_accuracy = diff_accuracy + float(row[0])  
    diff_accuracy = diff_accuracy / times


with open( "diff_precision_DNN.txt") as txtfile:
    rows = csv.reader(txtfile)
    diff_precision = 0 
    for row in rows:
        diff_precision = diff_precision + float(row[0])      
    diff_precision = diff_precision / times

with open( "diff_recall_DNN.txt") as txtfile:
    rows = csv.reader(txtfile)
    diff_recall = 0 
    for row in rows:
        diff_recall = diff_recall + float(row[0])   
    diff_recall = diff_recall / times


with open( "diff_metrice_DNN.txt") as txtfile:
    rows = csv.reader(txtfile)
    diff_TP = 0 
    diff_TN = 0 
    diff_FP = 0 
    diff_FN = 0 
    for row in rows:
        diff_TP = diff_TP + float(row[0])  
        diff_TN = diff_TN + float(row[1])  
        diff_FP = diff_FP + float(row[2])  
        diff_FN = diff_FN + float(row[3])  
    diff_TP = diff_TP /times  
    diff_TN = diff_TN /times   
    diff_FP = diff_FP/times  
    diff_FN = diff_FN /times   



fp = open("experiment_result.txt","a")
fp.write("Normal : \n")
fp.write("Acc: " + str(normal_accuracy) + "\n")
fp.write("Pre: " + str(normal_precision) + "\n")
fp.write("Rec: " + str(normal_recall) + "\n")
fp.write("TP : " + str(normal_TP) + "\n")
fp.write("TN : " + str(normal_TN) + "\n")
fp.write("FP : " + str(normal_FP) + "\n")
fp.write("FN : " + str(normal_FN) + "\n\n")
fp.write("Diff : \n")
fp.write("Acc: " + str(diff_accuracy) + "\n")
fp.write("Pre: " + str(diff_precision) + "\n")
fp.write("Rec: " + str(diff_recall) + "\n")
fp.write("TP : " + str(diff_TP) + "\n")
fp.write("TN : " + str(diff_TN) + "\n")
fp.write("FP : " + str(diff_FP) + "\n")
fp.write("FN : " + str(diff_FN) + "\n")
