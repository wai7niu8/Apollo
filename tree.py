import types
import copy_reg
import numpy as np
import multiprocessing

"""
according to Steven Bethard's recipe: https://bytes.com/topic/python/answers/552476-why-cant-you-pickle-instancemethods
use copy_reg to make our method picklable, because multiprocessing must pickle things to sling them among process.
"""
def _pickle_method(method):
    func_name = method.im_func.__name__
    obj = method.im_self
    cls = method.im_class
    return _unpickle_method, (func_name, obj, cls)

def _unpickle_method(func_name, obj, cls):
    for cls in cls.mro():
        try:
            func = cls.__dict__[func_name]
        except KeyError:
            pass
        else:
            break
    return func.__get__(obj, cls)

copy_reg.pickle(types.MethodType, _pickle_method, _unpickle_method)

class TreeNode(object):
    def __init__(self,is_leaf=False,leaf_score=None,factor=None,feature=None,left_child=None,right_child=None,nan_direction=0):
        """
        :param is_leaf: if True, only need to initialize leaf_score. other params are for intermediate tree node
        :param leaf_score: prediction score of the leaf node
        :param factor: split factor of the intermediate node
        :param feature: split feature of the intermediate node
        :param left_child: left child node
        :param right_child: right child node
        :param nan_direction: if 0, those NAN sample goes to left child, if 1 goes to right child.
                              goes to left child by default
        """
        self.is_leaf = is_leaf
        self.factor = factor
        self.feature = feature
        self.nan_direction = nan_direction
        self.left_child = left_child
        self.right_child = right_child
        self.leaf_score = leaf_score

class Tree(object):
    def __init__(self):
        self.root = None
        self.min_instances_byleaf = None
        self.reg_lambda = None
        self.gamma = None
        self.min_child_weight = None
        self.max_depth = 0
        self.pool_size = 1
        self.feature_importance = {}
        self.factors = []
        self.factors_type = {}

    def construct_tree(self, instances, factors, depth, factors_type):
        """
        Construct the current tree using BFS.
        """
        gradSum, hessSum, sorted_instances = self.presort(instances, factors)  #pre-sorted
        if len(instances) < self.min_instances_byleaf or depth > self.max_depth or hessSum < self.min_child_weight:
            """
            three return conditions:
            1,the number of instances of this node < min_instances_byleaf
            2,the depth of current tree achieved the max_depth
            3,the weight of this node < min_child_weight(according to xgboost, choosed hess as weight of the instance).
            """
            is_leaf = True
            leaf_score = self.compute_leaf_score(instances)
            return TreeNode(is_leaf=is_leaf, leaf_score=leaf_score)
        max_gain = -np.inf
        split_factor = factors[0]
        split_feature = 0.0
        nan_direction = True
        left_ids = []
        right_ids = []
        factors_sorted_infos = [[factor, factors_type[factor], sorted_instances[factor]] for factor in factors]

        pool=multiprocessing.Pool(processes=self.pool_size)

        ret_pools = pool.map(self.compute_split_feature, factors_sorted_infos)
        pool.close()
        for ret in ret_pools:
            if ret[0] > max_gain:
                max_gain = ret[0]
                split_factor = ret[1]
                split_feature = ret[2]
                nan_direction = ret[5]
                left_ids = ret[3]
                right_ids = ret[4]
                if split_factor in self.feature_importance:
                    self.feature_importance[split_factor] += 1
                else:
                    self.feature_importance[split_factor] = 1
        left_instances, right_instances = self.split_cur_instances(instances, left_ids, right_ids)
        left_tree = self.construct_tree(left_instances,factors, depth+1,factors_type)
        right_tree = self.construct_tree(right_instances,factors, depth+1,factors_type)
        return TreeNode(is_leaf=False,leaf_score=None,factor=split_factor,feature=split_feature,left_child=left_tree,right_child=right_tree,nan_direction=nan_direction)
    
    def compute_split_feature(self, factor_sorted_infos):
        """
        1,Select the feature threshold which brings the max gain
        2,Study the direction for missing value automatically
        
        :param factor_sorted_infos: [factor, factor_type, feature2idx],
                feature2idx: the sorted info(featureList, idxList...) of one factor
        :param numerical: the feature is numerical or not
        """
        factor, factor_type, feature2idx = factor_sorted_infos
        featureList, idxList, GradList, HessList, nanIdxList, GradSum, HessSum, Gnan, Hnan = feature2idx
        if not featureList:
            raise ValueError("featureList is null")
        nan_direction = True
        gain = -np.inf
        splitIndex = 0
        lastIndex = 0
        for index in range(len(featureList)-1):
            if featureList[index] == featureList[index+1]:
                continue
            #assign nan to left child
            if(factor_type):
                GL = sum(GradList[:index+1]) + Gnan
                HL = sum(HessList[:index+1]) + Hnan
            else:
                GL = sum(GradList[lastIndex:index+1]) + Gnan
                HL = sum(HessList[lastIndex:index+1]) + Hnan
            GR = GradSum - GL
            HR = HessSum - HL
            gainLeft = self.compute_split_gain(GradSum, HessSum, GL, GR, HL, HR, self.reg_lambda, self.gamma)
            #assign nan to right child
            if(factor_type): #the feature is numerical
                GL = sum(GradList[:index+1])
                HL = sum(HessList[:index+1])
            else:  #the feature is categorical
                GL = sum(GradList[lastIndex:index+1])
                HL = sum(HessList[lastIndex:index+1])
            GR = GradSum - GL
            HR = HessSum - HL
            gainRight = self.compute_split_gain(GradSum, HessSum, GL, GR, HL, HR, self.reg_lambda, self.gamma)
            if gain < gainLeft or gain < gainRight:
                splitIndex = index+1
                gain = max(gainLeft,gainRight)
                nan_direction = gainLeft > gainRight
            lastIndex = index + 1
        retLeftIds = idxList[:splitIndex]
        retRightIds = idxList[splitIndex:]
        if nan_direction:
            retLeftIds.extend(nanIdxList)
        else:
            retRightIds.extend(nanIdxList)
        return gain, factor, featureList[splitIndex], retLeftIds, retRightIds, nan_direction
    
    def compute_split_gain(self, GradSum, HessSum, GL, GR, HL, HR, reg_lambda, gamma):
        """
        According to xgboost's formula 7
        """
        return 0.5 * ((GL * GL) / (HL + reg_lambda) + (GR * GR) / (HR + reg_lambda) - (GradSum * GradSum) / (HessSum + reg_lambda)) - gamma

    def compute_leaf_score(self, instances):
        """
        According to xgboost, the leaf score is :
            - G / (H+lambda)
        """
        gradSum = 0
        hessSum = 0
        for idx in instances:
            gradSum += instances[idx]["grad"]
            hessSum += instances[idx]["hess"]
        return - gradSum/(hessSum+self.reg_lambda)

    def presort(self, instances, factors):
        """
        Presort the instances for every factor
        """
        gradSum = 0      #the sum of grad of all instances
        hessSum = 0      #the sum of hess of all instances
        factor2idx = {}
        for factor in factors:
            Gnan = 0    
            Hnan = 0    
            idx2feature = {}
            feature2idx = {"featureList":[], #sort according to current factor's value
                           "idxList":[],     #sort according to current factor's value
                           "gradList":[],    #sort according to current factor's value
                           "hessList":[],    #sort according to current factor's value
                           "nanIdxList":[],  #the row_idx of instance which current factor's value missing
                           "gradSum":0,
                           "hessSum":0,
                           "Gnan":0,         #the sum of grad of instance which current factor's value missing
                           "Hnan":0 }        #the sum of hess of instance which current factor's value missing
            saveOrder = ["featureList","idxList","gradList","hessList","nanIdxList","gradSum","hessSum","Hnan","Gnan"]
            for idx in instances:
                if factor == factors[0]:  #accumulated grad and hess only once
                    gradSum += instances[idx]["grad"]
                    hessSum += instances[idx]["hess"]
                if instances[idx][factor] == "nan":
                    Gnan += instances[idx]["grad"]
                    Hnan += instances[idx]["hess"]
                    feature2idx["nanIdxList"].append(idx)
                else:
                    idx2feature[idx] = instances[idx][factor]
            feature2idx["gradSum"] = gradSum
            feature2idx["hessSum"] = hessSum
            feature2idx["Hnan"] = Hnan
            feature2idx["Gnan"] = Gnan
            for (idx,feature) in sorted(idx2feature.iteritems(),key=lambda a:a[1]):
                feature2idx["featureList"].append(feature)
                feature2idx["idxList"].append(idx)
                feature2idx["gradList"].append(instances[idx]["grad"])
                feature2idx["hessList"].append(instances[idx]["hess"])
            assert len(feature2idx["featureList"]) == len(feature2idx["idxList"])
            factor2idx[factor] = [feature2idx[key] for key in saveOrder]
        return gradSum, hessSum, factor2idx

    def split_cur_instances(self, instances, left_ids, right_ids):
        """
        Split the current instances into two subinstances
        """
        left_instances = {}
        right_instances = {}
        for idx in instances:
            if idx in left_ids:
                left_instances.update({idx:instances[idx]})
            else:
                right_instances.update({idx:instances[idx]})
        return left_instances, right_instances

    def fit(self, instances, factors, factors_type, max_depth=6, pool_size=1, min_child_weight=1, min_instances_byleaf=10, reg_lambda=1.0, gamma=0.0):
        self.factors = factors
        self.factors_type = factors_type
        self.pool_size = pool_size
        self.max_depth = max_depth
        self.min_instances_byleaf = min_instances_byleaf
        self.reg_lambda = reg_lambda
        self.gamma = gamma
        self.min_child_weight = min_child_weight
        # build the tree by a recursive way
        self.root = self.construct_tree(instances, self.factors, 1, self.factors_type)

    def _predict(self, treenode, instance):
        """
        Predict a single instance
        """
        if treenode.is_leaf:
            return treenode.leaf_score
        elif instance[treenode.factor] == "nan":
            if treenode.nan_direction:
                return self._predict(treenode.left_child, instance)
            else:
                return self._predict(treenode.right_child, instance)
        elif instance[treenode.factor] < treenode.feature:
            return self._predict(treenode.left_child, instance)
        else:
            return self._predict(treenode.right_child, instance)

    def predict(self, instances):
        preds = {}
        for idx in instances:
            preds.update({idx:self._predict(self.root,instances[idx])})
        return preds


