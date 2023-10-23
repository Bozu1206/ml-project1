import numpy as np

def compute_accuracy(y_true, y_pred):
    matches = np.sum(y_true == y_pred)
    return matches / y_true.shape[0]
 
def f1_score(y_true, y_pred):
    
    assert len(y_true) == len(y_pred), "Length of actual and predicted values must be the same"
    
    TP = sum(1 for true, pred in zip(y_true, y_pred) if true == pred == 1)
    FP = sum(1 for true, pred in zip(y_true, y_pred) if true == -1 and pred == 1)
    FN = sum(1 for true, pred in zip(y_true, y_pred) if true == 1 and pred == -1)
    
    precision = TP / (TP + FP) if (TP + FP) != 0 else 0
    recall = TP / (TP + FN) if (TP + FN) != 0 else 0
    
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) != 0 else 0
    return f1

    