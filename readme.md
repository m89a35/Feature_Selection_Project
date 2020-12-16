這是 許嘉榮 於 2020/7 完成的 研究所碩論方向尋找時的研究。

動機與摘要 :

由於在執行深度學習之前，時常會遇到取得的資料集未經特徵選則的情況發生，導致有許多特徵其實與預測模型結果並無太大關連。但這些關聯性若並非該領域的專業人士，無法快速的了解並篩選，因此特徵選取就成了棘手的問題。該問題的解空間大小為 2 的 特徵數量次方，想透過暴力法來解決此問題時間成本過高。若透過人工進行試錯 (Trial and error) 會耗費過多的時間以及人力成本，且容易落入區域最佳解。本研究透過 Genetic algorithm (GA) 搭配KNN來執行特徵選取，研究發現，若未透過特徵選取，而將整個資料集皆餵入深度學習模型，100次的執行結果準確率只在 77.2%，若透過基因演算法來進行特徵選擇後的100次執行結果準確度則在 79.96%，實驗結果準確度上升了2.76%。

資料集 : NSL-KDD 資料集(入侵偵測系統)

參數設置 : 

    深度學習的超參數設置 :

        batch size : 2000 

        epoch : 40

        model = Sequential()

        hidden layer : 7

        hidden layer neuron number : 145 92 134 73 146 77 157 

        (via the last project : PSO tuning hyperparameters)

        hidden layer activation function : Relu

        output layer activation function : Softmax

        loss function : categorical_crossentropy
        
        optimizer : adam 

執行方式 : 

將整包皆放在同一個目錄底下，分為三個部分。
1.GA搭配KNN得到最佳子集合
    cmd執行 python GA_knn.py [iteration] [particle_num] [crossoer_rate] [mutation_rate] [player]
    會得到 update.txt ，第一欄為特徵數量，第二欄為KNN得到的準確度，第三欄為特徵子集合(0為不使用，1為使用)。

    解釋 : iteration 為程式要跑的迭代次數， particle_num 為每個iteration基因的數量， 
           crossover_rate 為基因交配的機率， mutation_rate 為基因突變的機率， 
           player 為touranment 方法進行比賽的選手數。

2.將該子集合資料集餵入DNN進行訓練並得到結果
    到DNN_train.py裡面更改欲訓練的feature_list(update.txt內的第三欄，可自行選擇與調整)
    也可更改batch size 與 epoch。
    sol.txt則可設定隱藏層層數以及神經元個數(第一欄為隱藏層層數，接著十欄為各層之神經元個數)。
    設定好後 cmd 執行 python DNN_train.py (可透過.sh執行多次，透過第三支程式做平均)
    會得到實驗結果( 準確率、精度、recall以及混淆矩陣)

3.得到N次的結果之平均
    進入average.py 設定times(方才跑DNN_train.py的次數)
    執行python average.py 
    得到experiment_result.txt為此次實驗之結果。


Dependency list 已放在 requirement.txt 裡。
