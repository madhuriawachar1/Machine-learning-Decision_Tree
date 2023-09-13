import pandas as pd
import numpy as np
import time
from tree.base import DecisionTree
from metrics import *
import seaborn as sns
import matplotlib.pyplot as plt

np.random.seed(42)
num_average_time = 100



# ...
# Function to plot the results 



def createFakeData(N,P,case):
    if(case==1):
        X = pd.DataFrame(np.random.randn(N, P))
        y = pd.Series(np.random.randn(N))
        
    elif(case==2):
        X = pd.DataFrame(np.random.randn(N, P))
        y = pd.Series(np.random.randint(2, size = N), dtype="category")
    elif(case==3):
        X = pd.DataFrame({i:pd.Series(np.random.randint(2, size = N), dtype="category") for i in range(5)})
        y = pd.Series(np.random.randint(2, size = N), dtype="category")
        #print('X\n',X)
        #print('y\n',y)
    else:
        X = pd.DataFrame({i:pd.Series(np.random.randint(2, size = N), dtype="category") for i in range(5)})
        y = pd.Series(np.random.randn(N))

    return X, y


def plotTimings(case,learning_time,predict_time):
    if(case==1):
        plt.plot(list(learning_time))
        plt.ylabel('RIRO : Fit time', fontsize=16)
        plt.show()

        plt.plot(list(predict_time))
        plt.ylabel('RIRO : Predict time', fontsize=16)
        plt.show()
    elif(case==2):
        plt.plot(list(learning_time))
        plt.ylabel('RIDO : Fit time', fontsize=16)
        plt.show()

        plt.plot(list(predict_time))
        plt.ylabel('RIDO : Predict time', fontsize=16)
        plt.show()

    elif(case==3):
        plt.plot(list(learning_time))
        plt.ylabel('DIRO : Fit time', fontsize=16)
        plt.show()
        
        plt.plot(list(predict_time))
        plt.ylabel('DIRO : Predict time', fontsize=16)
        plt.show()
    else:
        plt.plot(list(learning_time))
        plt.ylabel('DIDO : Fit time', fontsize=16)
        plt.show()

        plt.plot(list(predict_time))
        plt.ylabel('DIDO : Predict time', fontsize=16)
        plt.show()


def analyseTime(case):
    assert(1<=case<=4)
    learning_time = list()
    predict_time = list()

    for Ni in range(6,42):
        for step in range(1,7):
            N = Ni
            P = step
           
            X, y = createFakeData(N,P,case)   
            #print('X\n',X)
            #print('y\n',y)
            start_time = time.time()
            tree = DecisionTree(criterion="information_gain")
            tree.fit(X, y)
            end_time = time.time()
                
            learning_time.append(end_time-start_time)

            start_time = time.time()
            y_hat = tree.predict(X)
            end_time = time.time()
                
            predict_time.append(end_time-start_time)

    plotTimings(case,learning_time,predict_time)
    


analyseTime(4)
analyseTime(3)
analyseTime(2)
analyseTime(1)