import argparse
import os
from os import path

import torch
import torch.backends.cudnn as cudnn
#from torch.utils.tensorboard import SummaryWriter

#from features_loader import FeaturesLoader
#from network.TorchUtils import TorchModel
#from network.anomaly_detector_model import AnomalyDetector, custom_objective, RegularizedLoss
#from utils.callbacks import DefaultModelCallback, TensorBoardCallback
#from utils.utils import register_logger, get_torch_device

import h5py
import numpy as np


def get_args():
    parser = argparse.ArgumentParser(description="PyTorch Video Classification Parser")

  # io
    parser.add_argument('--features_path', dest='features_dir', default='features',
                        help="path to features")
    parser.add_argument('--annotation_path', default="Train_Annotation.txt",
                        help="path to train annotation")
    parser.add_argument('--log_file', type=str, default="log.log",
                        help="set logging file.")
    parser.add_argument('--exps_dir', type=str, default="exps",
                        help="set logging file.")
    parser.add_argument('--save_name', type=str, default="model",
                        help="name of the saved model.")
    parser.add_argument('--checkpoint', type=str,
                        help="load a model for resume training")

    """
    # optimization
    parser.add_argument('--batch_size', type=int, default=60,
                        help="batch size")
    parser.add_argument('--feature_dim', type=int, default=4096,
                        help="batch size")
    parser.add_argument('--save_every', type=int, default=20000,
                        help="epochs interval for saving the model checkpoints")
    parser.add_argument('--lr_base', type=float, default=0.01,
                        help="learning rate")
    parser.add_argument('--epochs', type=int, default=20000,
                        help="maxmium number of training epoch")
    """
    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()
    params = vars(args)
    features_dir = "C:/deep/Pytorch_C3D_Feature_Extractor/data/c3d_features/c3d_fc6_features.hdf5"  #params['features_dir']

    f = h5py.File(features_dir, "r")
    for videos in f.keys():
        print(videos)
        #print(np.array(f[videos]['c3d_features']))

    f.close()



