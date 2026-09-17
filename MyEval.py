import torch

class MyEval():
    def __init__(self):
        pass

    # value range of prediction: 0 ~ 1
    # value range of label: 0 or 1
    # accuracy range: 0 ~ 1 
    def compute_accuracy(self, prediction, label):
        threshold   = 0.5
        label_pred  = (prediction >= threshold)
        bCorrect    = (label_pred == label)
        accuracy    = bCorrect.sum() / len(label)
        return accuracy
        
