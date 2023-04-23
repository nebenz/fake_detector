import os
import torch
# from scipy.io.wavfile import write
import sys
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as stats
import hydra

sys.path.append(os.getcwd())
from src.data.data_utils import AudioPreProcessing,AudioUtills

from sklearn.metrics import roc_curve, auc
from scipy.optimize import brentq
from scipy.interpolate import interp1d


class Metrics():
    def __init__(self):
        super().__init__()


    
    @staticmethod
    def post_process(args,pred,target):
        target = target.to(pred.device)
        bias = float(target - pred)
        SE = np.square(bias)
        MAE = np.abs(bias)

        score = {'bias': bias, 'SE': SE, 'MAE': MAE}
        # pred = pred[:,0,:,:]+1j*pred[:,1,:,:]
        # if args.cut_shape_f:
        #     pred = torch.cat((pred[:,0,:].unsqueeze(1)*0,pred),axis=1)
        # pred = torch.istft(pred,window=torch.hamming_window(args.window_size).to(pred.device),hop_length=args.overlap, n_fft=args.window_size,onesided=True,center=False)
        return bias, SE,  MAE


    
    def calculate_eer(y_true, y_pred):
        if np.unique(y_pred).size == 1:
            fpr = np.array([0, 1])
            tpr = np.array([0, 1])
            thresholds = np.array([y_pred[0], y_pred[0]])
        else:
            fpr, tpr, thresholds = roc_curve(y_true, y_pred, pos_label=1)
        roc_auc = auc(fpr, tpr)
        eer = brentq(lambda x : 1. - x - interp1d(fpr, tpr)(x), 0., 1.)
        eer = torch.tensor(eer)
        return eer, fpr, tpr, roc_auc
   


def inference(pl_model, input):
    input = input.unsqueeze(0)
    input = input.to(pl_model.device)
    pred = pl_model(input)
    pred = (pred > 0.5).int().item()
    return pred
    # pred = pl_model.infer([input,target])
    # return pred

def pred_model(args,pl_model):

    # ============================= initialize =============================================
    
    orig_wd =hydra.utils.get_original_cwd()
    files_path = orig_wd + args.data_dir.test
    labels_path = orig_wd + args.data_label_test

    filelist = os.listdir(files_path)
    hash_table = AudioUtills.create_hash_table(labels_path)


    # device = torch.device('cuda:{:d}'.format(next(model.parameters()).device.index)) if torch.cuda.is_available() else 'cpu'

    audio_pre_process = AudioPreProcessing(args)
    metrics = Metrics()
    score = {'target':[], 'pred':[]}
    for i, filename in enumerate(filelist):

    # ============================= data pre process =============================================
        if filename.endswith('flac'):
            target = hash_table[filename]
            input_features, target = audio_pre_process.pre_processing(files_path + '/' + filename, target) 
            
            # ============================= inference =============================================
            input_features = torch.unsqueeze(input_features,0)
            pred = inference(pl_model, input_features)
            if pred == -1:
                continue

        # ============================= post process ==================================================

            
            score['target'].append(target)
            score['pred'].append(pred)
            

    eer, _,_,_ = Metrics.calculate_eer(score['target'], score['pred'])
    # ============================= evaluate metrics and save results =============================
    correct = 0
    for y_t, y_p in zip(score['pred'],  score['target']):
        if y_t == y_p:
            correct += 1
    
    # calculate the accuracy as the fraction of correct predictions
    accuracy = correct / len(score['target'])   
    
    return eer, accuracy

