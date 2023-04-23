import numpy as np
import hydra
import torch
#device = torch.device('cpu')
from scipy.io.wavfile import write
import os
import sys
# sys.path.append('your src folder')
from utills_model import pred_model
from pl_module import load_models




epsilon = 1e-6

# ======================================== main section ==================================================

Hydra_path = os.path.join(os.getcwd(), "src","config")
@hydra.main(config_path= Hydra_path,config_name="train.yaml")
def main(args):
    # ============================= create instance model with regular Pl_module =============================================
    Load_model  = load_models(args)
    os.environ["CUDA_VISIBLE_DEVICES"] = '0'
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
  

    model = Load_model.load(args.model_name)
    model.to(device)
    model.eval()
    # args.eval_mode = True
    eer, accuracy = pred_model(args,model)
    print("Accuracy of test is:", accuracy , ". EER is:", eer)
    # args.eval_mode = False

if __name__ == '__main__':
    main()

