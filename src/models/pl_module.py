import sys
import pytorch_lightning as pl
import random, os
import hydra

# sys.path.append('/workspace/inputs/asaf')
from architectures import Encoder_Classification , Simple_Classification, ResNetBinaryClassifier, UNet_R_with_transformer_mask_log, UNet_R_with_transformer_map_log



import numpy as np
import torch

def seed_everything(seed: int):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True
    

class load_pl_module(pl.LightningModule):
    def __init__(self, model):
        super().__init__()
        self.model = model
    def forward(self,x):
        pred = self.model(x)
        return pred

class load_models():
    def __init__(self,args):
        super().__init__()
        self.args = args
     

    @staticmethod
    def choose_model(args): 
       
       
        if args.model_name=='encoder_classification':
            return  Encoder_Classification(args)
        if args.model_name=='simple_classification':
            return  Simple_Classification(args)
        if args.model_name=='resNet_BinaryClassifier':
            return  ResNetBinaryClassifier(args)
        if args.model_name=='UNet_R_with_transformer_mask_log':
            return  UNet_R_with_transformer_mask_log(args)
        if args.model_name=='UNet_R_with_transformer_map_log':
            return  UNet_R_with_transformer_map_log(args)
       
 
    def load(self,model_def_name):
        model_type = self.choose_model(self.args)
        model = load_pl_module(model_type)
        model = model.load_from_checkpoint(model=model_type,checkpoint_path =hydra.utils.get_original_cwd() +  self.args.infer_from_ckpt)
        return model
