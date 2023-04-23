import os
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl
import hydra

from src.data.data_utils import AudioUtills, AudioPreProcessing
import warnings

warnings.filterwarnings("ignore", message="The version_base parameter is not specified")



class GeneretedInputOutput(Dataset):
    """Generated Input and Output to the model"""
    def __init__(self,args,data_dir, data_label):
        """ 
        Args:
            data_dir (string): Directory with all the files.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.args = args
        self.data_dir = hydra.utils.get_original_cwd() + data_dir
        self.data_label = hydra.utils.get_original_cwd() + data_label
        hash_table = AudioUtills.create_hash_table(self.data_label)
        mix_wav_files = []
        label_arr = []
        for root, dirs, files in os.walk(self.data_dir):
            for file in files:
                if ".flac" in file:
                    mix_wav_files.append(os.path.join(root, file))
                    label_arr.append(hash_table[file])


        self.os_listdir = mix_wav_files
        self.labels = label_arr

       
        self.audio_pre_process = AudioPreProcessing(self.args)

    def __len__(self):
        return len(self.os_listdir)

    def __getitem__(self,idx):
        # ======== get audio file name ====================
        file_path = self.os_listdir[idx]
        label =  self.labels[idx]

        # ==================== data-preprocessing ======================
        input_features, target = self.audio_pre_process.pre_processing(file_path, label)
        return input_features, target


class DataloaderModule(pl.LightningDataModule):
    def __init__(self,args): 
        super().__init__()
        self.args=args

    def setup(self,stage=None):   
        self.train_set = GeneretedInputOutput(self.args,self.args.data_dir.train, self.args.data_label_train)
        self.val_set = GeneretedInputOutput(self.args,self.args.data_dir.val, self.args.data_label_eval)
        self.test_set = GeneretedInputOutput(self.args,self.args.data_dir.test, self.args.data_label_test)

    def train_dataloader(self):
        return DataLoader(self.train_set,batch_size=self.args.train_batch_size, shuffle=True , num_workers =self.args.num_workers, pin_memory= self.args.pin_memory)

    def val_dataloader(self):
        return DataLoader(self.val_set,batch_size=self.args.val_batch_size, shuffle= False, num_workers = self.args.num_workers, pin_memory= self.args.pin_memory)

    def test_dataloader(self):
        return DataLoader(self.test_set,batch_size=self.args.test_batch_size, shuffle= True, num_workers = self.args.num_workers, pin_memory= self.args.pin_memory)
            


    