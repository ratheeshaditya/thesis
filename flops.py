 #!/bin/env python
 
'''
import packages
'''

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch._six import inf
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from functools import partial
from random import shuffle
from sklearn.model_selection import KFold
from prettytable import PrettyTable
from mlflow import log_metric, log_param, start_run
from datetime import timedelta
from einops import rearrange
import time
import pickle
import sys
import csv
import yaml
import numpy as np
import pandas as pd
import os
import logging
import argparse
from thop import profile
from torchsummary import summary


from trajectory import Trajectory, TrajectoryDataset, extract_fixed_sized_segments, split_into_train_and_test, remove_short_trajectories, get_categories, get_UTK_categories, get_NTU_categories
from transformer import TubeletTemporalSpatialPart_concat_chan_2_Transformer, TubeletTemporalPart_concat_chan_1_Transformer, TubeletTemporalTransformer, TubeletTemporalPart_mean_chan_1_Transformer, TubeletTemporalPart_mean_chan_2_Transformer, TubeletTemporalPart_concat_chan_2_Transformer, TemporalTransformer_4, TemporalTransformer_3, TemporalTransformer_2, BodyPartTransformer, SpatialTemporalTransformer, TemporalTransformer, Block, Attention, Mlp
from utils import print_statistics, SetupLogger, evaluate_all, evaluate_category, conv_to_float, SetupFolders, train_acc

# logger.info("Reading args")

begin_time = time.time() # Log starting time

parser = argparse.ArgumentParser()
parser.add_argument("--config_file", help="file from which configs need to be loaded", default="config")
args = parser.parse_args()

with open(args.config_file, "r") as ymlfile:
    cfg = yaml.load(ymlfile, Loader=yaml.FullLoader)

base_folder, model_dir, log_dir, results_dir = SetupFolders(cfg['META']['NAME'], cfg['MODEL']['DATASET'])

logger = SetupLogger('logger', log_dir)
logger.info("Logger set up!")
logger.info("Tensorboard set up!\n\n\n\n")
logger.info("FOLDER NAME: %s", cfg['META']['NAME'])
logger.info("\nCONFIGS \n=======\n"+yaml.dump(cfg))

with open(os.path.join(base_folder,'config.yml'), 'w') as config_file:
    yaml.dump(cfg, config_file)

# with start_run(run_name=args.filename):
log_param("filename", cfg['META']['NAME'])
log_param("embed_dim", cfg['MODEL']['EMBED_DIM'])
log_param("debug", cfg['MODEL']['DEBUG'])
log_param("epochs", cfg['TRAINING']['EPOCHS'])
log_param("patience", cfg['TRAINING']['PATIENCE'])
log_param("k_fold", cfg['TRAINING']['KFOLD'])
log_param("lr", cfg['TRAINING']['LR'])
log_param("lr_patience", cfg['TRAINING']['LR_PATIENCE'])
log_param("model_type", cfg['MODEL']['MODEL_TYPE'])
log_param("segment_length", cfg['MODEL']['SEGMENT_LEN'])
log_param("dataset", cfg['MODEL']['DATASET'])
log_param("batch_size", cfg['TRAINING']['BATCH_SIZE'])
log_param("decomposed", cfg['DECOMPOSED']['ENABLE'])
log_param("weight_decay", cfg['TRAINING']['WEIGHT_DECAY'])
log_param("pad_mode", cfg['TUBELET']['PAD_MODE'])
log_param("kernel", cfg['TUBELET']['KERNEL'])
log_param("stride", cfg['TUBELET']['STRIDE'])

logger.info('Number of arguments given: %s arguments.', str(len(sys.argv)))
logger.info('Arguments given: %s', ';'.join([str(x) for x in sys.argv]))

logger.info('parser args: %s', str(args))

logger.info('CUDA available: %s', str(torch.cuda.is_available()))
# device = "cuda:0" if torch.cuda.is_available() else "cpu"
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
logger.info('Available devices: %s', torch.cuda.device_count())
# logger.info('Current cuda device: %s ', str(torch.cuda.current_device()))

writer = SummaryWriter(log_dir=log_dir) # Tensorboard writer

if cfg['DECOMPOSED']['ENABLE']:
    if cfg['DECOMPOSED']['TYPE'] == "GR":
        dec_GR_path = "decom_GR_"
    elif cfg['DECOMPOSED']['TYPE'] == "GS":
        dec_GR_path = "decom_"

# Set dataset
dataset = cfg['MODEL']['DATASET']
wd = cfg['TRAINING']['WEIGHT_DECAY']

if dataset=="HRC":
    decomposed = dec_GR_path if cfg['DECOMPOSED']['ENABLE'] else ""
    dimension = "2D"

    PIK_train = "/home/s2435462/HRC/data/"+dataset+"/trajectories_train_HRC_"+decomposed+dimension+".dat"
    PIK_test = "/home/s2435462/HRC/data/"+dataset+"/trajectories_test_HRC_"+decomposed+dimension+".dat"

    all_categories = get_categories()
elif dataset == "UTK":
    PIK_train = "./data/train_UTK_trajectories.dat"
    PIK_test = "./data/test_UTK_trajectories.dat"
    all_categories = get_UTK_categories()
elif "NTU" in dataset:
    dimension = dataset.split('_')[-1]
    decomposed = dec_GR_path if cfg['DECOMPOSED']['ENABLE'] else ""

    PIK_train = "/home/s2435462/HRC/data/"+dataset+"/trajectories_train_NTU_"+decomposed+dimension+".dat"
    PIK_test = "/home/s2435462/HRC/data/"+dataset+"/trajectories_test_NTU_"+decomposed+dimension+".dat"
    # if "2D" in dataset:
    #     if cfg['TRAINING']['DECOMPOSED']:
    #         PIK_train = "/home/s2435462/HRC/data/trajectories_train_NTU_2D.dat"
    #         PIK_test = "/home/s2435462/HRC/data/trajectories_test_NTU_2D.dat"
    #     else:
    #         PIK_train = "/home/s2435462/HRC/data/trajectories_train_NTU_2D.dat"
    #         PIK_test = "/home/s2435462/HRC/data/trajectories_test_NTU_2D.dat"
    # elif "3D" in dataset:
    #     if cfg['TRAINING']['DECOMPOSED']:
    #         PIK_train = "/home/s2435462/HRC/data/trajectories_train_NTU_3D.dat"
    #         PIK_test = "/home/s2435462/HRC/data/trajectories_test_NTU_3D.dat"
    #     else:
    #         PIK_train = "/home/s2435462/HRC/data/trajectories_train_NTU_3D.dat"
    #         PIK_test = "/home/s2435462/HRC/data/trajectories_test_NTU_3D.dat"
    
    all_categories = get_NTU_categories()
else:
    raise Exception('dataset not recognized, must be HRC or NTU')



'''
LOAD TRAINING AND TEST DATASETS (ALREADY CREATED)
'''
logger.info("Loading train and test files")

with open(PIK_train, "rb") as f:
    train_crime_trajectories = pickle.load(f)

with open(PIK_test, "rb") as f:
    test_crime_trajectories = pickle.load(f)

logger.info("Loaded %d train and %d test files", len(train_crime_trajectories), len(test_crime_trajectories))

# Load the frame lengths to a list so that the min, max and mean no. of frames could be found
print_statistics(train_crime_trajectories, test_crime_trajectories, logger)



'''
SET THE SEGMENT LENGTH AND REMOVE SHORT TRAJECTORIES
'''
# Set the segment size
segment_length = cfg['MODEL']['SEGMENT_LEN']
# Remove the short trajectories from both train & test datasets
logger.info("Removing short trajectories")
train_crime_trajectories = remove_short_trajectories(train_crime_trajectories, input_length=segment_length, input_gap=0, pred_length=0)
test_crime_trajectories = remove_short_trajectories(test_crime_trajectories, input_length=segment_length, input_gap=0, pred_length=0)


'''
DEBUG MODE
'''
if cfg['MODEL']['DEBUG']:
    train_crime_trajectories = {key: value for key, value in train_crime_trajectories.items() if 'S001' in key or 'S002' in key or 'Shooting001' in key or 'Arson002'}
    test_crime_trajectories = {key: value for key, value in test_crime_trajectories.items() if 'S003' in key or 'S004' in key or 'RoadAccidents010' in key or 'Robbery002'}  
    logger.info('IN DEBUG MODE!!!\n')  


logger.info("Categories: %s", ','.join(all_categories))

model_name = cfg['META']['NAME'] #e.g. "transformer_model_embed_dim_32"
embed_dim = cfg['MODEL']['EMBED_DIM']

file_name_train = os.path.join(results_dir, 'training.csv')
file_name_test = os.path.join(results_dir, 'testing.csv')



'''
TRAINING CODE
'''
logger.info("STARTING TRAINING")

def train_model(embed_dim, epochs):

    # Set batch size
    batch_size = cfg['TRAINING']['BATCH_SIZE']
    
    # prepare cross validation

    n = cfg['TRAINING']['KFOLD']
    
    logger.info("Applying K-Fold with k = %d", n) 
    kf = KFold(n_splits=n, random_state=42, shuffle=True)
    
    #file to save results
    if dataset == "HRC":
        num_classes = 13
        num_joints = 17
        num_parts = 5
        in_chans = 2
    elif dataset == "UTK":
        num_classes = 10
        num_joints = 20
        in_chans = 3
    elif "NTU" in dataset:
        if "2D" in dataset:
            num_classes = 120
            num_joints = 25
            num_parts = 5
            in_chans = 2
        elif "3D" in dataset:
            num_classes = 120
            num_joints = 25
            num_parts = 5
            in_chans = 3

    if cfg['DECOMPOSED']['ENABLE']:
        if cfg['DECOMPOSED']['TYPE'] == "GR":
            num_joints = num_joints*2
        elif cfg['DECOMPOSED']['TYPE'] == "GS":
            num_joints+=1
    
    with open(file_name_train, 'a') as csv_file_train:
        csv_writer_train = csv.writer(csv_file_train, delimiter=';')
        csv_writer_train.writerow(['fold', 'epoch', 'LR', 'Training Loss', 'Validation Loss', 'Validation Accuracy', 'Time'])
    
    with open(file_name_test, 'a') as csv_file_test:
        csv_writer_test = csv.writer(csv_file_test, delimiter=';')
        csv_writer_test.writerow(['fold', 'label', 'video', 'person', 'prediction', 'log_likelihoods'])
    
        
    '''
    Load segments from the trajectories and create Dataset from them
    '''

    # segmented_path_train = '/home/s2435462/HRC/data/'+dataset+'/segmented/segmented_trajectory_train_'+cfg['MODEL']['DATASET']+'_'+str(cfg['MODEL']['SEGMENT_LEN'])+'_'+decomposed+str(cfg['MODEL']['DEBUG'])+'.pkl'
    # segmented_path_test = '/home/s2435462/HRC/data/'+dataset+'/segmented/segmented_trajectory_test_'+cfg['MODEL']['DATASET']+'_'+str(cfg['MODEL']['SEGMENT_LEN'])+'_'+decomposed+str(cfg['MODEL']['DEBUG'])+'.pkl'

    # if os.path.exists(segmented_path_train):
    #     logger.info("Loading segmented Trajectory train dataset from %s",segmented_path_train)
    #     with open(segmented_path_train, 'rb') as fo:
    #         train = pickle.load(fo)
    #     logger.info("Loading segmented Trajectory test dataset %s", segmented_path_test)
    #     with open(segmented_path_test, 'rb') as fo:
    #         test = pickle.load(fo)
    # else:
    #     logger.info("Creating Trajectory Train and Test datasets")
    #     train = TrajectoryDataset(*extract_fixed_sized_segments(dataset, train_crime_trajectories, input_length=segment_length))
    #     test = TrajectoryDataset(*extract_fixed_sized_segments(dataset, test_crime_trajectories, input_length=segment_length))
    #     logger.info("Writing segmented Trajectory train dataset to %s", segmented_path_train)
    #     with open(segmented_path_train, 'wb') as fi:
    #         pickle.dump(train, fi)
    #     logger.info("Writing segmented Trajectory test dataset to %s", segmented_path_test)
    #     with open(segmented_path_test, 'wb') as fi:
    #         pickle.dump(test, fi)

    logger.info("Creating Trajectory Train and Test datasets")
    train = TrajectoryDataset(*extract_fixed_sized_segments(dataset, train_crime_trajectories, input_length=segment_length))
    test = TrajectoryDataset(*extract_fixed_sized_segments(dataset, test_crime_trajectories, input_length=segment_length))


    def collator_for_lists(batch):
        '''
        Reference : https://stackoverflow.com/questions/64883998/pytorch-dataloader-shows-odd-behavior-with-string-dataset
        Reference : https://stackoverflow.com/questions/52818145/why-pytorch-dataloader-behaves-differently-on-numpy-array-and-list
        '''
        # assert all('sentences' in x for x in batch)
        # assert all('label' in x for x in batch)
        return {
            'id': [x['id'] for x in batch],
            'videos': [x['videos'] for x in batch],
            'persons': [x['persons'] for x in batch],
            'frames': torch.tensor(np.array([x['frames'] for x in batch])),
            'categories': torch.tensor(np.array([x['categories'] for x in batch])),
            'coordinates': torch.tensor(np.array([x['coordinates'] for x in batch]))
        }

    logger.info('--------------------------------')

    logger.info('No. of trajectories to train: %s', len(train_crime_trajectories))
    
    logger.info("Starting K-Fold")

    # K-fold Cross Validation model evaluation
    for fold, (train_ids, val_ids) in enumerate(kf.split(train.trajectory_ids()), 1):
        logger.info('\nfold: %d, train: %d, test: %d', fold, len(train_ids), len(val_ids))

        logger.info("Creating Train and Validation subsets.")

        train_subset = torch.utils.data.Subset(train, train_ids)
        val_subset = torch.utils.data.Subset(train, val_ids)

        logger.info("Creating Train and Validation dataloaders.")

        train_dataloader = torch.utils.data.DataLoader(train_subset, batch_size = batch_size, shuffle=True, collate_fn=collator_for_lists)
        val_dataloader = torch.utils.data.DataLoader(val_subset, batch_size = batch_size, shuffle=True, collate_fn=collator_for_lists)

        logger.info("Creating the model.")
        #intialize model
        if cfg['MODEL']['MODEL_TYPE'] == 'temporal':
            model = TemporalTransformer(embed_dim=embed_dim, num_frames=segment_length, num_classes=num_classes, num_joints=num_joints, in_chans=in_chans, mlp_ratio=2., qkv_bias=True, qk_scale=None, dropout=0.1)
        elif cfg['MODEL']['MODEL_TYPE'] == 'temporal_2':
            model = TemporalTransformer_2(embed_dim=embed_dim, num_frames=segment_length, num_classes=num_classes, num_joints=num_joints, in_chans=in_chans, mlp_ratio=2., qkv_bias=True, qk_scale=None, dropout=0.1)
        elif cfg['MODEL']['MODEL_TYPE'] == 'temporal_3':
            model = TemporalTransformer_3(embed_dim=embed_dim, num_frames=segment_length, num_classes=num_classes, num_joints=num_joints, num_parts=num_parts, in_chans=in_chans, mlp_ratio=2., qkv_bias=True, qk_scale=None, dropout=0.1)
        elif cfg['MODEL']['MODEL_TYPE'] == 'temporal_4':
            model = TemporalTransformer_4(embed_dim=embed_dim, num_frames=segment_length, num_classes=num_classes, num_joints=num_joints, num_parts=num_parts, in_chans=in_chans, mlp_ratio=2., qkv_bias=True, qk_scale=None, dropout=0.1)
        elif cfg['MODEL']['MODEL_TYPE'] == 'spatial-temporal':
            model = SpatialTemporalTransformer(embed_dim_ratio=embed_dim, num_frames=segment_length, num_classes=num_classes, num_joints=num_joints, in_chans=in_chans, mlp_ratio=2., qkv_bias=True, qk_scale=None, dropout=0.1)
        elif cfg['MODEL']['MODEL_TYPE'] == "parts":
            model = BodyPartTransformer(dataset=dataset, embed_dim_ratio=embed_dim, num_frames=segment_length, num_classes=num_classes, num_joints=num_joints, in_chans=in_chans, mlp_ratio=2., qkv_bias=True, qk_scale=None, dropout=0.1)
        elif cfg['MODEL']['MODEL_TYPE'] == "tubelet_temporal":
            kernel = tuple(map(int, cfg['TUBELET']['KERNEL'].split(',')))
            stride = tuple(map(int, cfg['TUBELET']['STRIDE'].split(',')))
            model = TubeletTemporalTransformer(dataset=dataset, embed_dim=embed_dim, num_frames=segment_length, num_classes=num_classes, num_joints=num_joints, in_chans=in_chans, kernel=kernel, stride=stride, mlp_ratio=2., qkv_bias=True, qk_scale=None, dropout=0.1)
        elif cfg['MODEL']['MODEL_TYPE'] == "ttpmc1":
            kernel = tuple(map(int, cfg['TUBELET']['KERNEL'].split(',')))
            stride = tuple(map(int, cfg['TUBELET']['STRIDE'].split(',')))
            model = TubeletTemporalPart_mean_chan_1_Transformer(dataset=dataset, embed_dim=embed_dim, num_frames=segment_length, num_classes=num_classes, num_joints=num_joints, in_chans=in_chans, kernel=kernel, stride=stride, mlp_ratio=2., qkv_bias=True, qk_scale=None, dropout=0.1)
        elif cfg['MODEL']['MODEL_TYPE'] == "ttpcc1":
            kernel = tuple(map(int, cfg['TUBELET']['KERNEL'].split(',')))
            stride = tuple(map(int, cfg['TUBELET']['STRIDE'].split(',')))
            model = TubeletTemporalPart_concat_chan_1_Transformer(dataset=dataset, embed_dim=embed_dim, num_frames=segment_length, num_classes=num_classes, num_joints=num_joints, in_chans=in_chans, kernel=kernel, stride=stride, mlp_ratio=2., qkv_bias=True, qk_scale=None, dropout=0.1)
        elif cfg['MODEL']['MODEL_TYPE'] == "ttpmc2":
            kernel = tuple(map(int, cfg['TUBELET']['KERNEL'].split(',')))
            stride = tuple(map(int, cfg['TUBELET']['STRIDE'].split(',')))
            model = TubeletTemporalPart_mean_chan_2_Transformer(dataset=dataset, embed_dim=embed_dim, num_frames=segment_length, num_classes=num_classes, num_joints=num_joints, in_chans=in_chans, kernel=kernel, stride=stride, mlp_ratio=2., qkv_bias=True, qk_scale=None, dropout=0.1)
        elif cfg['MODEL']['MODEL_TYPE'] == "ttpcc2":
            kernel = tuple(map(int, cfg['TUBELET']['KERNEL'].split(',')))
            stride = tuple(map(int, cfg['TUBELET']['STRIDE'].split(',')))
            model = TubeletTemporalPart_concat_chan_2_Transformer(dataset=dataset, embed_dim=embed_dim, num_frames=segment_length, num_classes=num_classes, num_joints=num_joints, in_chans=in_chans, kernel=kernel, stride=stride, mlp_ratio=2., qkv_bias=True, qk_scale=None, dropout=0.1)
        elif cfg['MODEL']['MODEL_TYPE'] == "ttspcc2":
            kernel = tuple(map(int, cfg['TUBELET']['KERNEL'].split(',')))
            stride = tuple(map(int, cfg['TUBELET']['STRIDE'].split(',')))
            model = TubeletTemporalSpatialPart_concat_chan_2_Transformer(dataset=dataset, embed_dim_ratio=embed_dim, num_frames=segment_length, num_classes=num_classes, num_joints=num_joints, in_chans=in_chans, kernel=kernel, stride=stride, mlp_ratio=2., qkv_bias=True, qk_scale=None, dropout=0.1, pad_mode = cfg['TUBELET']['PAD_MODE'])
        
        else:
            raise Exception('model_type is missing, must be temporal, temporal_2, temporal_3, temporal_4, spatial-temporal or parts')
        
        model.to(device)

        logger.info("Models defined")
        
        # Initialize parameters with Glorot / fan_avg.
        # This code is very important! It initialises the parameters with a
        # range of values that stops the signal fading or getting too big.
        for p in model.parameters():
            if p.dim() > 1:
                #print('parameter:',p)
                nn.init.xavier_uniform_(p)
        
        # Define optimizer
        optim = torch.optim.Adam(model.parameters(), lr=cfg['TRAINING']['LR'], weight_decay=wd, betas=(0.9, 0.98), eps=1e-9)
        cross_entropy_loss = nn.CrossEntropyLoss()
        
        '''
        Define scheduler for adaptive learning
        learning rate patience < early stopping patience
        '''
        lr_patience = cfg['TRAINING']['LR_PATIENCE']
        scheduler = ReduceLROnPlateau(optim, patience = lr_patience, verbose=True) 
            
        # Early stopping parameters
        min_loss = inf
        patience =  cfg['TRAINING']['PATIENCE']
        logger.info('Early stopping patience: %d', patience)
        trigger_times = 0
            
        
        #print('start looping over epochs at', time.time())
        best_epoch = -1


        model.train()

        batch = next(iter(train_dataloader))
    
        data =  batch['coordinates']
        
        data = data.to(device)

        macs, params = profile(model, inputs=(data, ))

        logger.info("MACS: %d", macs)
        logger.info("PARAMS: %d", params)

        print(summary(model, data.shape[1:]))


        break
                

            
    

  

#train model
train_model(embed_dim=cfg['MODEL']['EMBED_DIM'], epochs=cfg['TRAINING']['EPOCHS'])

