import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tree.base import DecisionTree
from metrics import *
from sklearn.model_selection import train_test_split
np.random.seed(42)

# dataset read 
# -m pip install -U scikit-learns for python3
from sklearn.datasets import make_classification
X, y = make_classification(
n_features=2, n_redundant=0, n_informative=2, random_state=1, n_clusters_per_class=2, class_sep=0.5)
#print('x\n',X)
#print('y\n',y)    
 # For plotting
import matplotlib.pyplot as plt
#print('x[:0]\n',X[:,1])
plt.scatter(X[:, 0], X[:, 1], c=y)
print(X,y)
Xdata = pd.DataFrame(data=X)
ydata = pd.Series(data=y, dtype="category")


# train test define
split_tt = int(0.7*len(X))

X = Xdata.iloc[:split_tt, :]
Xtest = Xdata.iloc[split_tt:, :]
y = ydata.iloc[:split_tt]
ytest = ydata.iloc[split_tt:]

# test and training
for criteria in ['information_gain', 'gini_index']:
    tree = DecisionTree(criterion=criteria, max_depth=3)
    # Build Decision Tree
    tree.fit(X, y)
    #Predict
    ypre = tree.predict(X)
    ytestpre = tree.predict(Xtest)
    tree.plot()
    print('Criteria :', criteria)
    print('Train Accuracy: ', accuracy(ypre, y))
    print('Test Accuracy: ', accuracy(ytestpre, ytest))
    # Precesion and Recall for each class
    for cls in y.unique():
        print("Class =",cls)
        print('Precision: ', precision(ytestpre, ytest, cls))
        print('Recall: ', recall(ytestpre, ytest, cls))




# k(=5) fold cross-validation
'''def five_fold_validation(X,y,depth=5):
    """Function to do five fold cross validation on iris"""
    X_original = X
    y_original = y

    accs = []

    # last 5th chunk as test data
    clf = DecisionTree(criterion="information_gain",max_depth=depth)
    clf.fit(pd.DataFrame(X[0:85]),pd.Series(y[0:85],dtype = "category"))
    y_hat = clf.predict(pd.DataFrame(X[68:]))
    accs.append(accuracy(pd.Series(y_hat),pd.Series(y[68:])))

    # 4rd chunk as test data
    clf = DecisionTree(criterion="information_gain",max_depth=depth)
    pass_X = pd.DataFrame(np.append(X[51:],X[0:34],axis=0))
    pass_y = pd.Series(np.append(y[51:],y[0:34],axis=0), dtype="category")
    clf.fit(pass_X , pass_y)
    y_hat = clf.predict(pd.DataFrame(X[34:51]))
    accs.append(accuracy(pd.Series(y_hat),pd.Series(y[34:51])))
    
    # 3nd chunk as test data
    clf = DecisionTree(criterion="information_gain",max_depth=depth)
    clf.fit(pd.DataFrame(np.append(X[68:],X[0:51],axis=0)), pd.Series(np.append(y[68:],y[0:51],axis=0),dtype="category"))
    y_hat = clf.predict(pd.DataFrame(X[51:68]))
    accs.append(accuracy(pd.Series(y_hat),pd.Series(y[51:68])))
    
    # 2st chunk as test data
    clf = DecisionTree(criterion="information_gain",max_depth=depth)
    clf.fit(pd.DataFrame(X[17:]), pd.Series(y[17:],dtype="category"))
    y_hat = clf.predict(pd.DataFrame(X[0:17]))
    accs.append(accuracy(pd.Series(y_hat),pd.Series(y[0:17])))
    
    # 1st chunk as test data
    clf = DecisionTree(criterion="information_gain",max_depth=depth)
    clf.fit(pd.DataFrame(np.append(X[0:17],X[34:],axis=0)), pd.Series(np.append(y[0:17],y[34:],axis=0),dtype="category"))
    y_hat = clf.predict(pd.DataFrame(X[17:34]))
    accs.append(accuracy(pd.Series(y_hat),pd.Series(y[17:34])))

    print("Individual Accuracies:")
    print(*accs)
    print("Average Accuracy:")
    avg = sum(accs)/5
    print(avg)'''

acc = []
for i in range(5):
    len1 = int((i/5)*len(X))
    len2= int(((i+1)/5)*len(X))
    
    
    X = pd.concat([Xdata.iloc[:len1, :], Xdata.iloc[len2:,:]], ignore_index=True)
    Xtest = Xdata.iloc[len1:len2, :]
    y = pd.concat([ydata.iloc[:len1], ydata.iloc[len2:]], ignore_index=True)
    ytest = ydata.iloc[len1:len2]
    tree = DecisionTree(criterion="information_gain", max_depth=3)
    tree.fit(X, y)
    ytestpre = tree.predict(Xtest)
    acc.append(accuracy(ytestpre, ytest))

print("5 fold cross-validation average accuracy:", sum(acc)/5)



# optimal depth using nested cross validation

valaccuracyarray = list()               

no_of_outer_folds = 5                   
no_of_inner_folds = 7                   

acc = -1
optimaldepth = -1
for i in range(no_of_outer_folds):
    len1 = int((i/no_of_outer_folds)*len(X))
    len2 = int(((i+1)/no_of_outer_folds)*len(X))
    
    
    X = pd.concat([Xdata.iloc[:len1, :], Xdata.iloc[len2:,:]], ignore_index=True)
    Xtest = Xdata.iloc[len1:len2, :]
    
    y = pd.concat([ydata.iloc[:len1], ydata.iloc[len2:]], ignore_index=True)
    ytest = ydata.iloc[len1:len2]
    
    hepler_var = {"accuracy": -1, "depth":-1}
    val_accuracy = list()
    for depth in range(10):
        temp = 0
        for j in range(no_of_inner_folds):
            len1 = int((j/no_of_inner_folds)*X.shape[0])
            len2 = int(((j+1)/no_of_inner_folds)*X.shape[0])
            X_val = X.iloc[len1:len2, :]
            Xtrain = pd.concat([X.iloc[:len1, :], X.iloc[len2:, :]],ignore_index=True)
            y_val = y.iloc[len1:len2]
            ytrain = pd.concat([y.iloc[:len1], y.iloc[len2:]],ignore_index=True)
            tree = DecisionTree(criterion="information_gain", max_depth=depth)
            tree.fit(Xtrain, ytrain)
            y_val_hat = tree.predict(X_val)
            temp += accuracy(y_val_hat, y_val)
        temp = temp/no_of_inner_folds
        if(hepler_var["accuracy"]==-1):
            hepler_var["accuracy"] = temp
            hepler_var["depth"] = depth
        else:
            if(temp>hepler_var["accuracy"]):
                hepler_var["accuracy"] = temp
                hepler_var["depth"] = depth
        val_accuracy.append(temp)
    valaccuracyarray.append(val_accuracy)
    tree = DecisionTree(criterion="information_gain", max_depth=hepler_var["depth"])
    tree.fit(X,y)
    ytestpre = tree.predict(Xtest)
    accur = accuracy(ytestpre, ytest)
    
    if(accur>acc):
        acc = accur
        optimaldepth = hepler_var["depth"]
    print("Accuracy:",accur, "depth:", hepler_var["depth"])



print("Optimal Depth:", optimaldepth)               # Optimal Depth for best result show




# Plot results of 5 folds

deptharray = [[i for i in range(10)] for j in range(no_of_outer_folds)]

fig = plt.figure()
ax = plt.subplot()
for i in range(no_of_outer_folds):
    ax.plot(deptharray[i], valaccuracyarray[i])
ax.set_xlabel('depth')
ax.set_ylabel('accuracy')
plt.show()

#print(len(X))
