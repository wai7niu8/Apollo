What is Apollo:

On July 20, 1969, Apollo 11 Mission Commander Neil Armstrong became the first man to walk on the moon. He said, "that's one small step for man; one giant leap for mankind." And, my "Apollo" is a simple version of Tianqi Chen's xgboost, that's my one small step for Machine Learning. Small, but important.

Apollo Supports:

    1,Built-in loss, Square error loss for regression task, Logistic loss for classification task
    
    2,Early stopping, evaluate on validation set and conduct early stopping.
    
    3,Multi-processing, when finding the best factor and feature 
    
    4,Categorical feature, support it directly, not need to convert it to numbers
    
    5,Handle missing value, the tree can learn a direction for those with NAN feature value
    
    6,Feature importance, output the feature importance after training
    
    7,Regularization, lambda, gamma (as in xgboost scoring function)
    
    8,Randomness, subsample，colsample_bytree, like Random Forest

Data Format:
    
    Like the data(train_data, val_data, test_data) in data/..
    
    1,The data should have a headline consisted of all feature names and "label".
    
    2,The label should be the last column
    
    3,Separate the columns with commas
    
    4,Don't support sparse input(if some features missed, u should use "" to fill the position).

Using:
    
    Reference to the run.py in examples/..

Next：
    
    1,Improve the storage structure. Apollo only uses the built-in structure, which demage the efficiency.
    
    2,Improve the speed. Apollo uses pre-sorted while LightGBM uses histgram which is faster than the former.
    
    3,Support more kinds of problems, like multi-classification, etc.
    
    4,Implement more metrics, like F-score, etc.

Reference:
    
    XGBoost: A Scalable Tree Boosting System
    
    GREEDY FUNCTION APPROXIMATION: A GRADIENT BOOSTING MACHINE


