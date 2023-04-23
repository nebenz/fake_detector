import os
import sys
import hydra
import torch
import torch.nn as nn
import pytorch_lightning as pl
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.strategies import DDPStrategy
import hydra
import torch.nn.functional as F 
from utills_model import Metrics
import matplotlib.pyplot as plt
# import warnings
import numpy as np


sys.path.append(os.getcwd())
# sys.path.append('/Users/Asaf/Documents/fake_detector_v1/src')
from src.data.Dataloader import  DataloaderModule
import glob
from pl_module import load_models
from utills_model import pred_model
# seed_everything(42)

CRITERION = 'nn.criterion'

# ======================== Model section ===========================

class Pl_module(pl.LightningModule):
    def __init__(self,args):
        super().__init__()
        self.args = args
        self.model = load_models.choose_model(args)
        self.criterion = getattr(nn, self.args.criterion)(reduction= self.args.criterion_paramter_value)
        self.validation_step_outputs_loss = []
        self.validation_step_outputs_accuracy = []
        self.validation_step_outputs_eer = []
        self.y = []
        self.pred = []
       

    def forward(self,batch): 
        y_hat = self.model(batch)
        return y_hat
    
    def infer(self, batch):
        x, _ = batch
        x = torch.unsqueeze(torch.unsqueeze(x,0),0)
        x = x.to(self.device)     
        y_hat = self(x)
        pred = (y_hat > 0.5).float()

        
        return pred

    def training_step(self, batch, batch_idx):
        x, y = batch
        x = x.unsqueeze(1)
        y = y.float().unsqueeze(1)

        y_hat = self(x)
        
         
        loss = self.Loss(y_hat,y) # accepted is a 1d vector with dim [#batch]
        weight = torch.tensor(self.args.weights_loss).to(loss.device) #take care of the lack of 0 category
       
        weighted_loss = torch.mean(weight[y.long()].T.squeeze(0) * loss)
        self.log('train_loss', weighted_loss)
        return weighted_loss
        
    def validation_step(self, batch, batch_idx,loader_idx=None):
        x, y = batch
        
        x = x.unsqueeze(1)
        y = y.float().unsqueeze(1)
        y_hat = self(x)
        loss = self.Loss(y_hat,y)
        weight = torch.tensor(self.args.weights_loss).to(loss.device)
        weighted_loss = torch.mean(weight[y.long()].T.squeeze(0) * loss)
   
        pred = (y_hat > 0.5).float()
        acc = (pred == y).float().mean()
        y = y.cpu().numpy()
        pred = pred.cpu().numpy()
        eer, _, _, _ = Metrics.calculate_eer(y, pred)

        self.log('val_loss', weighted_loss, on_step = False, on_epoch=True)
        self.log('val_accuracy', acc, on_step = False, on_epoch=True)
        self.log('val_eer', eer, on_step = False, on_epoch=True)

        

        self.validation_step_outputs_loss.append(weighted_loss)
        self.validation_step_outputs_accuracy.append(acc)
        self.validation_step_outputs_eer.append(eer)
        self.y.append(y[:])
        self.pred.append(pred[:])

        return loss

    
    def on_validation_epoch_end(self, loss):
        
        epoch_average_loss = torch.stack(self.validation_step_outputs_loss, dim=0).mean()
        epoch_average_acc = torch.stack(self.validation_step_outputs_accuracy, dim=0).mean()
        epoch_average_eer = torch.stack(self.validation_step_outputs_eer, dim=0).mean()

        
        self.log("validation_epoch_end_loss_average", epoch_average_loss)
        self.log("validation_epoch_end_acc_average", epoch_average_acc)
        self.log("validation_epoch_end_eer_average", epoch_average_eer)

        _, fpr, tpr, roc_auc = Metrics.calculate_eer(np.concatenate(self.y), np.concatenate(self.pred))
        # Plot ROC curve
        fig, ax = plt.subplots()
        ax.plot(fpr, tpr, label=f"AUC = {roc_auc:.3f}")
        ax.plot([0, 1], [0, 1], "k--")
        ax.set_xlabel("False positive rate")
        ax.set_ylabel("True positive rate")
        ax.set_title("ROC curve")
        ax.legend(loc="lower right")


        # Log ROC curve plot using Lightning's logger
        self.logger.experiment.add_figure("ROC Curve Valid Epoch End", fig, self.current_epoch)



        self.validation_step_outputs_loss.clear()  # free memory
        self.validation_step_outputs_accuracy.clear()
        self.validation_step_outputs_eer.clear()
        self.y.clear()
        self.pred.clear()

       
        # self.args.is_validation_epoch_end = True
        eer, accuracy = pred_model(self.args, self)
        # eer, accuracy = eer.to(self.device)#,accuracy.to(self.device)
        self.log("Test: eer  ",eer)
        self.log("Test: accuracy  ",accuracy)    

        # self.args.is_validation_epoch_end = False

    def test_step(self, batch, batch_idx):
        pred = self(batch)
        loss = self.Loss(pred,y)
        self.log('test_loss', loss)

    def configure_optimizers(self):

        
        optimizer = torch.optim.Adam(self.parameters(), lr=self.args.learning_rate)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min',factor=0.9, patience=10, cooldown=3, verbose=True)
        return {'optimizer': optimizer,'scheduler':scheduler}

    def Loss(self,pred,y, parameters = None):

        loss = self.criterion(pred.squeeze(1),y.squeeze(1))
       
        return loss 

   
# ======================================== main section ==================================================

def find_existing_ckpt():

    folder_name = os.getcwd()
    ckpts = glob.glob(folder_name + "/**/last.ckpt", recursive=True)
    if len(ckpts) == 0:
        print("No checkpoints found.")
        return None
    times = [os.path.getctime(ckpt) for ckpt in ckpts]
    max_time = max(times)
    latest_index = times.index(max_time)
    print(f"Resuming from checkpoint: {ckpts[latest_index]}")
    return ckpts[latest_index]



Hydra_path = os.path.join(os.getcwd(), "src","config")
 # INTERACTIVE
# @hydra.main(config_path= Hydra_path,config_name="train.yaml")

# TRAIN
@hydra.main(config_path= __file__.split("src/")[0]+"src/config/",config_name="train.yaml", version_base='1.1')



def main(args):

# ====================== load config params from hydra ======================================

    pl_checkpoints_path = os.getcwd() + '/'
    # pl_checkpoints_path = args.models_dir + '/'


    if args.debug_flag: # debug mode
        a=1
   
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.cuda_visible_devices)

# ============================= main section =============================================

    model = Pl_module(args) #load_models( )
    # model.load(args.model_name)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # 'cuda'
    model= model.to(device)

    existing_ckpt = find_existing_ckpt()
    print(existing_ckpt)
    if existing_ckpt:
        model = model.load_from_checkpoint(
            model=model,
            checkpoint_path=existing_ckpt, args=args)
    
  
    checkpoint_callback = ModelCheckpoint(
        monitor = args.ckpt_monitor,
        filename='{epoch:02d}-{val_loss:.2f}',
        save_last = args.save_last,
        save_top_k = args.save_top_k,
        mode='min',
        verbose=True,
    )

    earlystopping_callback = EarlyStopping(monitor = args.ckpt_monitor, 
                                            patience = args.patience)

    trainer = Trainer(  
                        gpus=1, #args.gpus, 
                        accelerator = 'gpu',
                        strategy = DDPStrategy(find_unused_parameters=False),
                        fast_dev_run=False, 
                        check_val_every_n_epoch=args.check_val_every_n_epoch, 
                        default_root_dir= pl_checkpoints_path,                       
                        callbacks=[earlystopping_callback, checkpoint_callback], 
                        precision=32,
                        num_sanity_val_steps = 0,
                     )

    data_module = DataloaderModule(args)
    trainer.fit(model = model, datamodule = data_module)
    checkpoint_callback.best_model_path

if __name__ == '__main__':
    main()
