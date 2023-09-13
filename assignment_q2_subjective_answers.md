PART a:
With max-depth = 3 and 70:30 split into train and test, the results are as follows:

Criteria : information_gain Train Accuracy: 0.9285714285714286 Test Accuracy: 0.8666666666666667 Class = 1 Precision: 0.85 Recall: 0.9444444444444444 Class = 0  Precision: 0.9 Recall: 0.75 

Criteria : gini_index Train Accuracy:  0.9285714285714286 Test Accuracy: 0.8666666666666667 Class = 1 Precision: 0.85 Recall: 0.9411764705882353 Class = 0 Precision: 0.9 Recall: 0.75 Class = 0 Precision: 0.9 Recall: 0.75

PART b:
Using 5 fold cross-validation: Max Depth = 2 Average Accuracy: 0.9268907563025209

Optimum Depth using nested cross-validation = 2
no_of_outer_folds = 5 ,no_of_inner_folds = 7 ,Accuracy: 0.9375
Optimal Depth: 2


output:
5 fold cross-validation average accuracy: 0.9268907563025209
Accuracy: 0.9375 depth: 2
Accuracy: 0.8235294117647058 depth: 1
Accuracy: 0.9375 depth: 1
Accuracy: 0.7647058823529411 depth: 2
Accuracy: 0.9411764705882353 depth: 2
Optimal Depth: 2
Plots of accuracy vs depth of the 5 folds can be found in figure_1 and figure_2