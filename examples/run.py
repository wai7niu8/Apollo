import numpy as np
from apollo import Apollo
from data import Data

#read data
train_data = Data("data/train_data")
val_data = Data("data/val_data")
test_data = Data("data/test_data")

params = {'loss': "logisticloss",
          'eta': 0.3,
          'max_depth': 6,
          'pool_size': 1,
          'num_round': 500,
          'scale_pos_weight': 1.0,
          'subsample': 0.8,
          'colsample_bytree': 0.8,
          'min_instances_byleaf': 10,
          'min_child_weight': 5,
          'reg_lambda': 1,
          'gamma': 0,
          'eval_metric': "error",
          'early_stopping_rounds': 20}

#init
gbm = Apollo()
#train
gbm.fit(train_data, validation_data=val_data, early_stopping_rounds=params["early_stopping_rounds"],eval_metric=params["eval_metric"],
    loss=params['loss'], eta=params["eta"], num_round=params["num_round"], max_depth=params["max_depth"],pool_size=params["pool_size"],
    min_instances_byleaf=params['min_instances_byleaf'],scale_pos_weight=params["scale_pos_weight"], subsample=params["subsample"], 
    colsample_bytree=params['colsample_bytree'],min_child_weight=params['min_child_weight'], reg_lambda=params['reg_lambda'], gamma=params['gamma'])
#test
gbm.predict(test_data.getData(),result_file="result") #save predict result in "result".
