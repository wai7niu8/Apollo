import numpy as np

"""
just implemented two loss functions, "SquareLoss" for Regression and "LogisticLoss" for Binary Classification.
"""

class SquareLoss(object):
    """
    For Regression
    """
    def __init__(self, reg_lambda=0.0):
        self.reg_lambda = reg_lambda
    
    def transform(self,pred):
        return pred

    def grad(self, preds, labels):
        return preds - labels

    def hess(self, preds, labels):
        return np.ones_like(labels)


class LogisticLoss(object):
    """
    For Binary Classification
    """
    def __init__(self, reg_lambda=0.0):
        self.reg_lambda = reg_lambda
    
    def transform(self, preds):
        return 1.0/(1.0+np.exp(-preds))

    def grad(self, preds, labels):
        preds = self.transform(preds)
        return (1-labels)/(1-preds) - labels/preds

    def hess(self, preds, labels):
        preds = self.transform(preds)
        return labels/np.square(preds) + (1-labels)/np.square(1-preds)
