# fake detector

attached is a project that classify between real and fake speech

all parameters can be modified by train.yaml config file

there are few available architectures, see the config file

there must be a file in the /lightning_logs/version_0/checkpoints/ caled: 'best=0.44.ckpt',  it can be downloaded at:

https://www.dropbox.com/s/qspq1lldm3m16ma/best%3D0.44.ckpt?dl=0


inference can be done through: /fake_detector_v1/src/models/predict_model.py


to train the model add data from the ASVspoof2019 DB into the data folder.




