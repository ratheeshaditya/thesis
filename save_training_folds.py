 #!/bin/env python
 
 #import packages
import numpy as np
#import torch
#import torch.nn as nn
#from functools import partial
import time
import pickle
#from torch.utils.data import DataLoader
import sys
#import csv
from sklearn.model_selection import KFold
#from torch.optim.lr_scheduler import ReduceLROnPlateau
#from torch._six import inf
#import torch.nn.functional as F
import os
from statistics import mean


from trajectory import Trajectory, extract_fixed_sized_segments, remove_short_trajectories, get_categories, get_UTK_categories
#from transformer import TemporalTransformer_4, TemporalTransformer_3, TemporalTransformer_2, BodyPartTransformer, SpatialTemporalTransformer, TemporalTransformer, Block, Attention, Mlp

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--filename", help="filename to store trained model and results")
parser.add_argument("--embed_dim", help="embedding dimension used by the model", type=int)
parser.add_argument("--debug", help="load subset of trajectories in debug mode", action="store_true", default=False)
parser.add_argument("--epochs", help="maximum number of epochs during training", default=1000, type=int)
parser.add_argument("--patience", help="patience before early stopping is enabled", default=5, type=int)
parser.add_argument("--k_fold", help="number of folds used for cross-validation", default=3, type=int)
parser.add_argument("--lr", help="starting learning rate for adaptive learning", default=0.001, type=float)
parser.add_argument("--lr_patience", help="patience before learning rate is decreased", default=3, type=int)
parser.add_argument("--model_type", help="type of model to train, temporal, temporal_2, temporal_3, temporal_4, spatial-temporal or parts", type=str)
parser.add_argument("--segment_length", help="length of sliding window", default=12, type=int)
parser.add_argument("--dataset", help="dataset used HR-Crime or UTK", default="HR-Crime", type=str)

args = parser.parse_args()

print('Number of arguments given:', len(sys.argv), 'arguments.')
print('Arguments given:', str(sys.argv))

print('parser args:', args)

#sys.exit()


#Load test trajectories
dataset = args.dataset
if dataset=="HR-Crime":
    PIK_train = "./data/train_anomaly_trajectories.dat"
    #PIK_test = "./data/test_anomaly_trajectories.dat"
    all_categories = get_categories()
elif dataset == "UTK":
    PIK_train = "./data/train_UTK_trajectories.dat"
    #PIK_test = "./data/test_UTK_trajectories.dat"
    all_categories = get_UTK_categories()
else:
    raise Exception('dataset not recognized, must be HR-Crime or UTK')

with open(PIK_train, "rb") as f:
    train_crime_trajectories = pickle.load(f)


train_frame_lengths = []
#test_frame_lengths = []

for key in train_crime_trajectories:
    #print(key, ' : ', train_crime_trajectories[key])
    num_of_frames = len(train_crime_trajectories[key])
    #print('number of frames:', num_of_frames)
    
    train_frame_lengths.append(num_of_frames)

print('\nTRAIN minimum:', min(train_frame_lengths))
print('TRAIN maximum:', max(train_frame_lengths))

print('TRAIN mean:', mean(train_frame_lengths))

train_smaller_than_mean = [x for x in train_frame_lengths if x <= mean(train_frame_lengths)]

print('\nTRAIN smaller_than_mean:', len(train_smaller_than_mean))


#set segment size
segment_length = args.segment_length
train_crime_trajectories = remove_short_trajectories(train_crime_trajectories, input_length=segment_length, input_gap=0, pred_length=0)

#use subset for when debugging to speed things up, comment  out to train on entire training set
if args.debug:
    train_crime_trajectories = {key: value for key, value in train_crime_trajectories.items() if key[-8:] < '005_0005'}
    print('\nin debugging mode: %d train trajectories' % (len(train_crime_trajectories)))    
else:
    print('\nRemoved short trajectories: %d train trajectories left' % (len(train_crime_trajectories)))


print("\ncategories", all_categories)



#time.sleep(30) # Sleep for 30 seconds to generate memory usage in Peregrine


n = args.k_fold
    
    
print("Apply K-Fold with k = ", n) 
kf = KFold(n_splits=n, random_state=42, shuffle=True)

'''
#file to save results
if dataset == "HR-Crime":
    file_name_train = '/data/s3447707/MasterThesis/training_results/' + model_name + '.csv'
    file_name_test = '/data/s3447707/MasterThesis/testing_results/' + model_name + '.csv'
    num_classes = 13
    num_joints = 17
    num_parts = 5
    in_chans = 2
elif dataset == "UTK":
    file_name_train = '/data/s3447707/MasterThesis/UTK_training_results/' + model_name + '.csv'
    file_name_test = '/data/s3447707/MasterThesis/UTK_testing_results/' + model_name + '.csv'
    num_classes = 10
    num_joints = 20
    in_chans = 3
'''
        

traj_ids_train, traj_videos_train, traj_persons_train, traj_frames_train, traj_categories_train, X_train = extract_fixed_sized_segments(dataset, train_crime_trajectories, input_length=segment_length)

#print(traj_ids_train)

#print(type(kf.split(traj_ids_train)))
#print(kf.split(traj_ids_train))

# K-fold Cross Validation model evaluation
for fold, (train_ids, val_ids) in enumerate(kf.split(traj_ids_train), 1):
    print('\nfold: %s, train: %s, test: %s' % (fold, len(train_ids), len(val_ids)))
    
    print(type(train_ids))
    print(train_ids)
    
    filename_train_ids = '/data/s3447707/MasterThesis/kfold_splits/' + dataset + '_segment_length_' + str(segment_length) + '_kfold_' + str(n) + '_split_fold_' + str(fold) + '_train_ids.npy'
    filename_val_ids = '/data/s3447707/MasterThesis/kfold_splits/' + dataset + '_segment_length_' + str(segment_length) + '_kfold_' + str(n) + '_split_fold_' + str(fold) + '_val_ids.npy'
    
    with open(filename_train_ids, 'wb') as f:
        np.save(f, train_ids)       
    
    with open(filename_val_ids, 'wb') as f:
        np.save(f, val_ids)     
        
    print("Saved train_ids of fold {} to {}".format(fold, filename_train_ids))
    print("Saved val_ids of fold {} to {}".format(fold, filename_val_ids))
