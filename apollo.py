__author__ = 'WangYi'

from collections import defaultdict
import random
import numpy as np

from data import Data
from tree import Tree
from loss import SquareLoss, LogisticLoss
from metric import get_metric


class Apollo(object):
    """
    My one small step for Machine Learning !
    """
    def __init__(self):
        self.trees = []
        self.eta = None
        self.num_round = None
        self.loss = None
        self.max_depth = None
        self.subsample = None
        self.colsample_bytree = None
        self.min_instances_byleaf = None
        self.reg_lambda = None
        self.gamma = None
        self.min_child_weight = None
        self.scale_pos_weight = None
        self.first_round_pred = None
        self.feature_importance = defaultdict(lambda: 0)

    def fit(self, train_data, validation_data,  early_stopping_rounds=np.inf, eval_metric=None,
            loss="logisticloss", eta=0.3, num_round=1000, max_depth=6, pool_size=1,
            min_instances_byleaf=1,scale_pos_weight=1, subsample=0.8, colsample_bytree=0.8,
            min_child_weight=1, reg_lambda=1.0, gamma=0):

        """
        :param train_data: Data object, train data
        :param validation_data: Data object, validation data
        :param eta: learning rate
        :param num_round: number of boosting round
        :param max_depth: max depth of each tree
        :param pool_size: the num of processes
        :param subsample: row sample rate when building a tree
        :param colsample_bytree: column sample rate when building a tree
        :param min_instances_byleaf: min number of samples in a leaf node
        :param loss: loss object
                     logisticloss,squareloss
        :param reg_lambda: lambda
        :param gamma: gamma
        :param eval_metric: evaluation metric, provided: "accuracy"
        """
        self.eta = eta
        self.num_round = num_round
        self.max_depth = max_depth
        self.pool_size = pool_size
        self.subsample = subsample
        self.colsample_bytree = colsample_bytree
        self.reg_lambda = reg_lambda
        self.gamma = gamma
        self.min_instances_byleaf = min_instances_byleaf
        self.eval_metric = eval_metric
        self.min_child_weight = min_child_weight
        self.scale_pos_weight = scale_pos_weight
        self.first_round_pred = 0.0


        # initial loss function
        if loss == "logisticloss":
            self.loss = LogisticLoss(self.reg_lambda)
        elif loss == "squareloss":
            self.loss = SquareLoss(self.reg_lambda)
            self.first_round_pred = train_data.getLabelMean()
        else:
            raise NotImplementedError("loss should be 'logisticloss' or 'squareloss'")


        # to evaluate on validation set and conduct early stopping
        do_validation = True
        valData = validation_data.getData()
        if not valData:
            raise ValueError("validation_data is empty !")
        
        valIdxList = [] #save an fixed order
        valLabels  = [] 
        for idx in valData:
            valData[idx]['yPred'] = self.first_round_pred  #init it with traindata
            valIdxList.append(idx)
            valLabels.append(valData[idx]['label'])

        best_val_metric = np.inf
        best_round = 0
        become_worse_round = 0

        data = train_data.getData()
        if not train_data:
            raise ValueError("train_data is empty !")
        idxList = [] #save an fixed order
        labels = []
        for idx in data:
            data[idx]['yPred'] = self.first_round_pred
            data[idx]['grad'] = self.loss.grad(data[idx]['grad'], data[idx]['label'])
            data[idx]['hess'] = self.loss.hess(data[idx]['hess'], data[idx]['label'])
            if data[idx]['label'] == 1.0:
                data[idx]['weight'] = self.scale_pos_weight
            idxList.append(idx)
            labels.append(data[idx]['label'])
        labels = np.array(labels)
        for i in range(self.num_round):
            # weighted grad and hess
            for idx in data:
                data[idx]['grad'] = data[idx]['grad'] * data[idx]['weight']
                data[idx]['hess'] = data[idx]['hess'] * data[idx]['weight']

            # row and column sample before training the current tree
            factors = train_data.getFactors()
            factorTypes = train_data.getFeatureTypes()
            sampledFactors = random.sample(factors,int(len(factors)*self.colsample_bytree))
            sampledData = {}
            for idx in random.sample(idxList, int(len(idxList)*self.subsample)):
                sampledData.update({idx:data[idx]})

            # train current tree
            tree = Tree()
            tree.fit(sampledData, sampledFactors, factorTypes, max_depth=self.max_depth, pool_size=self.pool_size, min_child_weight=self.min_child_weight,
                     min_instances_byleaf=self.min_instances_byleaf, reg_lambda=self.reg_lambda, gamma=self.gamma)

            # predict the whole trainset and update y_pred,grad,hess
            preds = tree.predict(sampledData)
            for idx in sampledData:
                data[idx]['yPred'] += self.eta * preds[idx]
                data[idx]['grad'] = self.loss.grad(data[idx]["yPred"],data[idx]["label"])
                data[idx]['hess'] = self.loss.hess(data[idx]["yPred"],data[idx]["label"])

            # update feature importance
            for k in tree.feature_importance.iterkeys():
                self.feature_importance[k] += tree.feature_importance[k]

            self.trees.append(tree)

            # print training information
            if self.eval_metric is None:
                print "Apollo round {iteration}".format(iteration=i)
            else:
                try:
                    mertric_func = get_metric(self.eval_metric)
                except:
                    raise NotImplementedError("The given eval_metric is not provided")

                curPreds = np.array([data[idx]["yPred"] for idx in idxList])
                train_metric = mertric_func(self.loss.transform(curPreds), labels)

                if not do_validation:
                    print "Apollo round {iteration}, train-{eval_metric} is {train_metric}".format(
                        iteration=i, eval_metric=self.eval_metric, train_metric=train_metric)
                else:
                    valPreds = tree.predict(valData)
                    for idx in valData:
                        valData[idx]['yPred'] += self.eta * valPreds[idx]
                    curValPreds = [valData[idx]['yPred'] for idx in valIdxList]
                    assert len(curValPreds) == len(valLabels)
                    val_metric = mertric_func(self.loss.transform(np.array(curValPreds)), np.array(valLabels))
                    print "Apollo round {iteration}, train-{eval_metric} is {train_metric}, val-{eval_metric} is {val_metric}".format(
                        iteration=i, eval_metric=self.eval_metric, train_metric=train_metric, val_metric=val_metric
                    )

                    # check if to early stop
                    if val_metric < best_val_metric:
                        best_val_metric = val_metric
                        best_round = i
                        become_worse_round = 0
                    else:
                        become_worse_round += 1
                    if become_worse_round > early_stopping_rounds:
                        print "Apollo training Stop, best round is {best_round}, best val-{eval_metric} is {best_val_metric}".format(
                            best_round=best_round, eval_metric=eval_metric, best_val_metric=best_val_metric
                        )
                        break

    def predict(self, instances, result_file=""):
        if not result_file:
            raise IOError("You should assign a file to save result!")
        assert len(self.trees) > 0
        idxList = sorted([idx for idx in instances])
        # TODO: actually the tree prediction can be parallel
        preds = np.zeros(len(idxList))
        preds += self.first_round_pred
        for tree in self.trees:
            curPreds = tree.predict(instances)
            preds += self.eta * np.array([curPreds[idx] for idx in idxList])
        ret = open(result_file,"w")
        for pred in preds:
            ret.write(str(pred) + "\n")
        ret.close()
