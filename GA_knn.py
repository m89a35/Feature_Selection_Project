import random
import numpy as np
import csv 
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics
from keras.utils import np_utils 
# Parameters setting 
iteration = int(sys.argv[1]) 
particle_num = int(sys.argv[2]) 
crossover_rate = float(sys.argv[3]) 
mutation_rate = float(sys.argv[4]) 
player = int(sys.argv[5]) 

# Initialization
particle_X = []
particle_current_fitness = []
best_sol = []
best_fitness = 0
parent_particle = []
child_particle = []

# Random get the particle_num solutions.
def Initialization() :
    global particle_X
    for i in range(particle_num) :
        temp = []
        for j in range(41) :
            temp.append(random.randint(0,1))
        particle_X.append(list(temp))

# Get the accuracy of knn and use to caculate the fitness.
def Get_knn_accuracy(sol_list) :
    index_list = []
    for i in range(len(sol_list)) :
        if sol_list[i] == 1 :
            index_list.append(i)

    with open( "2class_NSL-KDDTrain+.txt") as txtfile:
        rows = csv.reader(txtfile)
        train_data = []
        train_label = []
        for row in rows:
            temp = []
            for i in range(len(index_list)) :
                temp.append(row[index_list[i]])    
            train_data.append(temp)
            train_label.append(row[41])
    
    with open( "2class_NSL-KDDTest+.txt") as txtfile:
        rows = csv.reader(txtfile)
        test_data = []
        test_label = []
        for row in rows:
            temp = []
            for i in range(len(index_list)) :
                temp.append(row[index_list[i]])    
            test_data.append(temp)
            test_label.append(row[41])
    model = KNeighborsClassifier(n_neighbors=3, n_jobs=-1)
    train_label = np_utils.to_categorical(train_label)
    train_data = np.array(train_data)
    test_data = np.array(test_data)
    test_label = np_utils.to_categorical(test_label)
    model.fit(train_data,np.array(train_label))
    y = model.predict(test_data)
    score = metrics.accuracy_score(test_label,y)
    return score*100

# You can set the weight of feature number and accuracy to balance the time and the accuracy.
def Fitness_caculate(particle):
    global best_sol
    global best_fitness
    particle_fitness = []
    for i in range(len(particle)) :
        fitness = 0 
        num = 0 
        for j in range(41) :
            if particle[i][j] == 1 :
                num += 1 
        fitness = num*(-0.2) + Get_knn_accuracy(particle[i])
        particle_fitness.append(fitness)
        if best_fitness < fitness :
            best_sol = list(particle[i])
            best_fitness = fitness
            fp = open("update.txt","a")
            fp.write(str(num) + " " + str(best_fitness + num*0.2) + " " + str(best_sol) + "\n")
            fp.close()
        print(fitness)
    return particle_fitness

# Choose the better gene to parent_particle and do crossover.
def Selection() :
    global parent_particle
    parent_particle = []
    for i in range(particle_num) :
        first_particle = random.randint(0,particle_num-1)
        second_particle = random.randint(0,particle_num-1)
        if particle_current_fitness[first_particle] >= particle_current_fitness[second_particle] :
            parent_particle.append(particle_X[first_particle])
        else :
            parent_particle.append(particle_X[second_particle])

# Choose two gene and do crossover by one point crossover if the random Probability smaller than crossover rate.
def Crossover() :
    global child_particle
    child_particle = []
    for i in range(particle_num) :
        first_particle = parent_particle[random.randint(0,particle_num-1)]
        second_particle = parent_particle[random.randint(0,particle_num-1)]
        if random.uniform(0,1) < crossover_rate :
            temp1 = []
            temp2 = []
            cutpoint = random.randint(1,particle_num-2)
            for i in range(cutpoint) :
                temp1.append(first_particle[i])
                temp2.append(second_particle[i])
            for i in range(41-cutpoint) :
                temp1.append(second_particle[i+cutpoint])
                temp2.append(first_particle[i+cutpoint])
            child_particle.append(list(temp1)) 
            child_particle.append(list(temp2))
        else :
            child_particle.append(list(first_particle))
            child_particle.append(list(second_particle))

# Do random one point mutation if Probability smaller than mutation rate.
def Mutation() :
    global child_particle
    for i in range(len(child_particle)) :
        if random.uniform(0,1) < mutation_rate :
            child_particle[i][random.randint(0,40)] = ( child_particle[i][random.randint(0,40)] + 1 ) % 2 

# Do selection2 by tournament and get the better gene.
def Selection2() :
    global particle_X 
    particle_X = []
    global particle_current_fitness
    particle_current_fitness = []
    child_particle_fitness = Fitness_caculate(child_particle)
    for i in range(particle_num) :
        best = random.randint(0,len(child_particle)-1)
        for j in range(player) :
            index = random.randint(0,len(child_particle)-1)
            if child_particle_fitness[index] > child_particle_fitness[best] :
                best = index 
        particle_X.append(child_particle[best])
        particle_current_fitness.append(child_particle_fitness[best])


#main
Initialization()
particle_current_fitness = Fitness_caculate(particle_X)

for i in range(iteration) :
    Selection()
    Crossover()
    Mutation()
    Selection2()
    
    print(best_fitness)
