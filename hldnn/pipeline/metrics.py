import  numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score
import torch
def eval_ap(y_true, y_pred):
    '''
        compute Average Precision (AP) averaged across tasks
    '''

    ap_list = []

    for i in range(y_true.shape[1]):
        # AUC is only defined when there is at least one positive data.
        if np.sum(y_true[:, i] == 1) > 0 and np.sum(y_true[:, i] == 0) > 0:
            # ignore nan values
            is_labeled = y_true[:, i] == y_true[:, i]
            ap = average_precision_score(y_true[is_labeled, i],
                                         y_pred[is_labeled, i])

            ap_list.append(ap)

    if len(ap_list) == 0:
        raise RuntimeError(
            'No positively labeled data available. Cannot compute Average Precision.')

    return {'ap': sum(ap_list) / len(ap_list)}


def binary_MSE_loss(y_pred, y_true):
    return (torch.sum(torch.square(y_pred[y_true==1]-1))/(torch.sum(y_true==1)+1e-5) + torch.sum(torch.square(y_pred[y_true==0]-0))/(torch.sum(y_true==0)+1e-5))/2
    
def binary_MAE_loss(y_pred, y_true):
    return (torch.abs(torch.square(y_pred[y_true==1]-1))/(torch.sum(y_true==1)+1e-5) + torch.abs(torch.square(y_pred[y_true==0]-0))/(torch.sum(y_true==0)+1e-5))/2

def binary_int_precision(y_pred, y_true):
    return (torch.sum(torch.round(y_pred[y_true==1])==1)/(torch.sum(y_true==1)+1e-5) + torch.sum(torch.round(y_pred[y_true==0])==0)/(torch.sum(y_true==0)+1e-5))/2