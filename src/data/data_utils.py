import os
import torch
import torchaudio
import hydra
import warnings
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore", message="The version_base parameter is not specified")

# ==================== Hyper-Params =================================================================================
EPS = 1e-8
# ===========================================================================================================

class AudioPreProcessing():
    def __init__(self,args,device=None):
        self.args = args
        self.device = device
        self.audioUtills = AudioUtills()

 
    
    #the output of this function should be ready to be send to the model
    def pre_processing(self, file_name, target = -1):
        input_features    = -1
        signal_wav, sr = self.audioUtills.load_wav(self.args , file_name)

        ########### input pre-process ###################

       
            
        # check and fit utt length
        signal_wav = self.audioUtills.fit_utt_length(self.args,signal_wav)
        if not isinstance(signal_wav, torch.Tensor):
            return -1, -1
        else:
            #STFT
            signal_stft = self.audioUtills.get_stft(self.args,signal_wav)
            input_features = signal_stft
        
        if self.args.spec_augment:
            input_features = self.audioUtills.spec_aug(self.args, input_features)

            
        return input_features, target
        


class AudioUtills():
    def __init__(self):
        super().__init__()
        # self.args = args

    @staticmethod
    def load_wav(args,path_wav,channel=0):
        signal_wav, sr = torchaudio.load(path_wav)
        # signal_wav = signal_wav[0,:]
        if sr != args.sample_rate:
            signal_wav = torchaudio.transforms.Resample(sr,args.sample_rate)(signal_wav)
        return signal_wav , sr
    
    @staticmethod
    def save_wav(wav,sr,path_wav):
        wav_norm = wav/1.1/(torch.abs(wav).max()+EPS).unsqueeze(0)
        torchaudio.save(wav_norm, wav, sr)


    @staticmethod
    def spec_aug(args,spec):
        spec_length=spec.size()[-1]
        I=spec_length//args.spec_aug_time_jump
        K_full=args.overlap

        for i in range(I):
            x_temp=torch.randint(0,args.spec_aug_time_jump-args.N_max,size=(1,))
            y_index=torch.randint(0,K_full-args.K_max,size=(1,))
            patch_len=torch.randint(args.N_min,args.N_max,size=(1,))        
            patch_higth=torch.randint(args.K_min,args.K_max,size=(1,))
            x_index=i*args.spec_aug_time_jump+x_temp
            spec[y_index:y_index+patch_higth,x_index:x_index+patch_len]=0
        return spec

  

    @staticmethod
    def get_stft(args,signal_wav):

        ''' TODO: add
        1. mel spec
        2. partial frequencies ?
        3. stft windows type


        '''

        

        signal_stft = torch.stft(signal_wav,window=torch.hann_window(args.window_size),hop_length=args.hop_length, n_fft=args.window_size,return_complex=True,center=False).squeeze(0)
        signal_stft  =signal_stft.abs()
        
        return signal_stft


   
    
    
    #fit to signal to a fixed size (4 sec.)
    @staticmethod
    def fit_utt_length(args,signal_wav):

        
        fixed_length = args.sample_rate*args.signal_length
        signal_length = signal_wav.shape[1]
        if signal_length >= fixed_length:
            signal_wav = torch.unsqueeze(signal_wav[0, :fixed_length],dim=0)
        else:
            num_rep = fixed_length//signal_length + 1
            signal_wav = torch.cat([signal_wav.repeat(1, num_rep)[:, :fixed_length]], dim=1)


      
        
        return signal_wav

    @staticmethod
    def create_hash_table(file):
        label_map = {}
        with open(file, 'r') as f:
            for line in f:
                match = line.split()
                file = match[1] #file name
                label_string = match[-1]
                if label_string == 'bonafide':
                    label = 0
                else:
                    label = 1
                label_map[f"{file}.flac"] = label
        
        return label_map
    
   








        

