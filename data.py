
"""
1,read the train/validation/test data into dict
2,using "nan" to express mising value
3,using exception to distinguish the feature type
"""
class Data(object):
    def __init__(self, data_file):
        self.data = {}
        self.rows = 0
        self.labelMean = 0
        self.factors = []
        self.featureType = {}

        with open(data_file,'r') as data_f:
            for line in data_f:
                line_pieces = line.strip().split(",")
                if self.rows == 0: #the headline
                    self.factors = line_pieces[:-1]
                    self.rows+=1
                    continue
                curData = self.data.setdefault(self.rows,{})
                curData.update({"grad":0,"hess":0,"yPred":0,"weight":1.0})
                curData.update({"label":float(line_pieces[-1])}) #the last col is label
                self.labelMean += float(line_pieces[-1])
                assert len(self.factors) == len(line_pieces[:-1])
                for idx in range(len(self.factors)):
                    if line_pieces[idx] == "":
                        curData.update({self.factors[idx]:"nan"}) #using "nan" to express mising value
                        continue
                    else:
                        try:
                            feature = float(line_pieces[idx])
                            if self.factors[idx] not in self.featureType:
                                self.featureType[self.factors[idx]] = True
                        except:
                            feature = line_pieces[idx]
                            if self.factors[idx] not in self.featureType:
                                self.featureType[self.factors[idx]] = False
                    curData.update({self.factors[idx]:feature})
                self.rows+=1
        self.labelMean /= self.rows

    def getData(self):
        return self.data

    def getFactors(self):
        return self.factors

    def getFeatureTypes(self):
        return self.featureType

    def getLabelMean(self):
        """
        return to init first_round_pred
        """
        return self.labelMean


