# fake detector

Attached is a project that classifies between real and fake speech.

All parameters can be modified by train.yaml config file

there are few available architectures, see the config file

there must be a file in the /lightning_logs/version_0/checkpoints/ caled: 'best=0.44.ckpt',  it can be downloaded at:

https://www.dropbox.com/s/qspq1lldm3m16ma/best%3D0.44.ckpt?dl=0


inference can be done through: /fake_detector_v1/src/models/predict_model.py


to train the model add data from the ASVspoof2019 DB into the data folder.

I only created here a prototype, there is a lot of work that can be done,and especially:

1. data is imbalanced, I only took care of it in the loss function.
2. spectrogram parameters - mel etc' - I used only a simplified version.
3. Supplied architectures are pretty basic.




