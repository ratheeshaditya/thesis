 #!/bin/env python
 
 #import packages
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch._six import inf
from torch.utils.data import DataLoader
from functools import partial
from random import shuffle
from sklearn.model_selection import KFold
import time
import pickle
import sys
import csv
import numpy as np
import os
import logging
import argparse

from trajectory import Trajectory, TrajectoryDataset, extract_fixed_sized_segments, split_into_train_and_test, remove_short_trajectories, get_categories, get_UTK_categories, get_NTU_categories
from transformer import TemporalTransformer_4, TemporalTransformer_3, TemporalTransformer_2, BodyPartTransformer, SpatialTemporalTransformer, TemporalTransformer, Block, Attention, Mlp
from utils import print_statistics, SetupLogger

logger = SetupLogger('logger')

# logging.basicConfig(format='%(asctime)s %(message)s', level=logging.INFO)

logger.info("Reading args")
# print ("Before args: "+time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))

parser = argparse.ArgumentParser()
parser.add_argument("--filename", help="filename to store trained model and results")
parser.add_argument("--embed_dim", help="embedding dimension used by the model", type=int)
parser.add_argument("--debug", help="load subset of trajectories in debug mode", action="store_true", default=False)
parser.add_argument("--epochs", help="maximum number of epochs during training", default=1000, type=int)
parser.add_argument("--patience", help="patience before early stopping is enabled", default=5, type=int)
parser.add_argument("--k_fold", help="number of folds used for corss-validation", default=3, type=int)
parser.add_argument("--lr", help="starting learning rate for adaptive learning", default=0.001, type=float)
parser.add_argument("--lr_patience", help="patience before learning rate is decreased", default=3, type=int)
parser.add_argument("--model_type", help="type of model to train, temporal, temporal_2, temporal_3, temporal_4, spatial-temporal or parts", type=str)
parser.add_argument("--segment_length", help="length of sliding window", default=12, type=int)
parser.add_argument("--dataset", help="dataset used HR-Crime or UTK", default="HR-Crime", type=str)
parser.add_argument("--batch_size", help="batch size for training", default=100, type=int)

args = parser.parse_args()

logger.info('Number of arguments given: %s arguments.', str(len(sys.argv)))
logger.info('Arguments given: %s', ';'.join([str(x) for x in sys.argv]))

logger.info('parser args: %s', str(args))

logger.info('CUDA available: %s', str(torch.cuda.is_available()))
device = "cuda:0" if torch.cuda.is_available() else "cpu"
logger.info('Available devices: %s', torch.cuda.device_count())
logger.info('Current cuda device: %s ', str(torch.cuda.current_device()))


# Set dataset
dataset = args.dataset
if dataset=="HR-Crime":
    PIK_train = "./data/train_anomaly_trajectories.dat"
    PIK_test = "./data/test_anomaly_trajectories.dat"
    all_categories = get_categories()
elif dataset == "UTK":
    PIK_train = "./data/train_UTK_trajectories.dat"
    PIK_test = "./data/test_UTK_trajectories.dat"
    all_categories = get_UTK_categories()
elif "NTU" in dataset:
    if "2D" in dataset:
        PIK_train = "/home/s2435462/HRC/data/trajectories_train_NTU_2D.dat"
        PIK_test = "/home/s2435462/HRC/data/trajectories_test_NTU_2D.dat"
    elif "3D" in dataset:
        PIK_train = "/home/s2435462/HRC/data/trajectories_train_NTU_3D.dat"
        PIK_test = "/home/s2435462/HRC/data/trajectories_test_NTU_3D.dat"
    all_categories = get_NTU_categories()
else:
    raise Exception('dataset not recognized, must be HR-Crime or UTK')

logger.info("Loading train and test files")

with open(PIK_train, "rb") as f:
    train_crime_trajectories = pickle.load(f)

with open(PIK_test, "rb") as f:
    test_crime_trajectories = pickle.load(f)

logger.info("Loaded %d train and %d test files", len(train_crime_trajectories), len(test_crime_trajectories))

# Load the frame lengths to a list so that the min, max and mean no. of frames could be found
print_statistics(train_crime_trajectories, test_crime_trajectories, logger)


# Set the segment size
segment_length = args.segment_length
# Remove the short trajectories from both train & test datasets
logger.info("Removing short trajectories")
train_crime_trajectories = remove_short_trajectories(train_crime_trajectories, input_length=segment_length, input_gap=0, pred_length=0)
test_crime_trajectories = remove_short_trajectories(test_crime_trajectories, input_length=segment_length, input_gap=0, pred_length=0)

#use subset for when debugging to speed things up, comment  out to train on entire training set
# if args.debug:
#     #train_crime_trajectories = {key: value for key, value in train_crime_trajectories.items() if key < 'Arrest'}
#     train_crime_trajectories = {key: value for key, value in train_crime_trajectories.items() if key[-8:] < '005_0005'}
#     test_crime_trajectories = {key: value for key, value in test_crime_trajectories.items() if key[-8:] < '005_0005'}
#     print('\nin debugging mode: %d train trajectories and %d test trajectories' % (len(train_crime_trajectories), len(test_crime_trajectories)))    
# else:
#     print('\nRemoved short trajectories: %d train trajectories and %d test trajectories left' % (len(train_crime_trajectories), len(test_crime_trajectories)))

logger.info("Categories: %s", ','.join(all_categories))



#time.sleep(30) # Sleep for 30 seconds to generate memory usage in Peregrine

model_name = args.filename #e.g. "transformer_model_embed_dim_32"
embed_dim = args.embed_dim

logger.info("STARTING TRAINING")

def train_model(embed_dim, epochs):

    # Set batch size
    batch_size = args.batch_size
    
    # prepare cross validation

    n = args.k_fold
    
    logger.info("Applying K-Fold with k = %d", n) 
    kf = KFold(n_splits=n, random_state=42, shuffle=True)
    
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
    elif "NTU" in dataset:
        if "2D" in dataset:
            file_name_train = '/home/s2435462/HRC/results/NTU_2D/training/' + model_name + '.csv'
            file_name_test = '/home/s2435462/HRC/results/NTU_2D/testing/' + model_name + '.csv'
            num_classes = 120
            num_joints = 25
            in_chans = 2
        elif "3D" in dataset:
            file_name_train = '/home/s2435462/HRC/results/NTU_3D/training/' + model_name + '.csv'
            file_name_test = '/home/s2435462/HRC/results/NTU_3D/testing/' + model_name + '.csv'
            num_classes = 120
            num_joints = 25
            in_chans = 3
        
    '''
    Load segments from the trajectories

    traj_ids_train : The trajectory ids
    traj_videos_train : The video ids
    traj_persons_train : The person ids
    traj_frames_train : The frames of each trajectories
    traj_categories_train : The categories of each trajectories
    X_train : The actual data, the coordinates for each frames

    '''
    logger.info("Creating Trajectory Train and Test datasets")
    train = TrajectoryDataset(*extract_fixed_sized_segments(dataset, train_crime_trajectories, input_length=segment_length))
    test = TrajectoryDataset(*extract_fixed_sized_segments(dataset, test_crime_trajectories, input_length=segment_length))

    # traj_ids_train, traj_videos_train, traj_persons_train, traj_frames_train, traj_categories_train, X_train = extract_fixed_sized_segments(dataset, train_crime_trajectories, input_length=segment_length)
    # traj_ids_test, traj_videos_test, traj_persons_test, traj_frames_test, traj_categories_test, X_test = extract_fixed_sized_segments(dataset, test_crime_trajectories, input_length=segment_length)
    
    logger.info("Writing to training log file")
    with open(file_name_train, 'w') as csv_file_train:
        csv_writer_train = csv.writer(csv_file_train, delimiter=';')
        csv_writer_train.writerow(['fold', 'epoch', 'LR', 'Training Loss', 'Validation Loss', 'Validation Accuracy', 'Time'])
        # prepare to write testing results to a file
        logger.info("Writing to testing log file")
        with open(file_name_test, 'w') as csv_file_test:
            csv_writer_test = csv.writer(csv_file_test, delimiter=';')
            csv_writer_test.writerow(['fold', 'label', 'video', 'person', 'prediction', 'log_likelihoods', 'logits'])
 
            # Start print
            logger.info('--------------------------------')
    
            logger.info('No. of trajectories to train: %s', len(train_crime_trajectories))
            
            logger.info("Starting K-Fold")

            # K-fold Cross Validation model evaluation
            for fold, (train_ids, val_ids) in enumerate(kf.split(train.trajectory_ids()), 1):
                logger.info('\nfold: %d, train: %d, test: %d', fold, len(train_ids), len(val_ids))

                train_subset = torch.utils.data.Subset(train, train_ids)
                val_subset = torch.utils.data.Subset(train, val_ids)
    
                # train_dataloader = torch.utils.data.DataLoader([ [traj_categories_train[i], traj_videos_train[i], traj_persons_train[i], traj_frames_train[i], X_train[i]] for i in train_ids], shuffle=True, batch_size=100)
                # val_dataloader = torch.utils.data.DataLoader([ [traj_categories_train[i], traj_videos_train[i], traj_persons_train[i], traj_frames_train[i], X_train[i]] for i in val_ids], shuffle=True, batch_size=100)
                train_dataloader = torch.utils.data.DataLoader(train_subset, batch_size = batch_size, shuffle=True)
                val_dataloader = torch.utils.data.DataLoader(val_subset, batch_size = batch_size, shuffle=True)

                #intialize model
                if args.model_type == 'temporal':
                        model = TemporalTransformer(embed_dim=embed_dim, num_frames=segment_length, num_classes=num_classes, num_joints=num_joints, in_chans=in_chans, mlp_ratio=2., qkv_bias=True, qk_scale=None, dropout=0.1)
                elif args.model_type == 'temporal_2':
                        model = TemporalTransformer_2(embed_dim=embed_dim, num_frames=segment_length, num_classes=num_classes, num_joints=num_joints, in_chans=in_chans, mlp_ratio=2., qkv_bias=True, qk_scale=None, dropout=0.1)
                elif args.model_type == 'temporal_3':
                        model = TemporalTransformer_3(embed_dim=embed_dim, num_frames=segment_length, num_classes=num_classes, num_joints=num_joints, num_parts=num_parts, in_chans=in_chans, mlp_ratio=2., qkv_bias=True, qk_scale=None, dropout=0.1)
                elif args.model_type == 'temporal_4':
                        model = TemporalTransformer_4(embed_dim=embed_dim, num_frames=segment_length, num_classes=num_classes, num_joints=num_joints, num_parts=num_parts, in_chans=in_chans, mlp_ratio=2., qkv_bias=True, qk_scale=None, dropout=0.1)
                elif args.model_type == 'spatial-temporal':
                    model = SpatialTemporalTransformer(embed_dim_ratio=embed_dim, num_frames=segment_length, num_classes=num_classes, num_joints=num_joints, in_chans=in_chans, mlp_ratio=2., qkv_bias=True, qk_scale=None, dropout=0.1)
                elif args.model_type == "parts":
                    model = BodyPartTransformer(embed_dim_ratio=embed_dim, num_frames=segment_length, num_classes=num_classes, num_joints=num_joints, in_chans=in_chans, mlp_ratio=2., qkv_bias=True, qk_scale=None, dropout=0.1)
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
                optim = torch.optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.98), eps=1e-9)
                
                '''
                Define scheduler for adaptive learning
                learning rate patience < early stopping patience
                '''
                lr_patience = args.lr_patience
                scheduler = ReduceLROnPlateau(optim, patience = lr_patience, verbose=True) 
                  
                # Early stopping parameters
                min_loss = inf
                patience =  args.patience
                logger.info('Early stopping patience: %d', patience)
                trigger_times = 0
                  
                start = time.time()
                temp = start
                
                #print('start looping over epochs at', time.time())
                
                for epoch in range(1, epochs+1):
    
                    train_loss = 0.0
                    
                    model.train()

                    logger.info("Enumerating Train loader")
               
                    for iter, batch in enumerate(train_dataloader, 1):
                    
                        ids, videos, persons, frames, data, labels = batch
                        
                        labels = labels.to(device)
                        videos = videos#.to(device)
                        persons = persons#.to(device)
                        frames = frames.to(device)
                        data = data.to(device)
    
                        output = model(data)
                        #output.to(device)
                     
                        #print(f"output shape: {output.shape}")
                        #print("output", output)
    
                
                        #print(f"labels shape: {labels.shape}")
                        #print(f"videos shape: {videos.shape}")
                        #print(f"persons shape: {persons.shape}")
                        #print(f"frames shape: {frames.shape}")
                        #print(f"data shape: {data.shape}")
    
                        #print("labels", labels)
                        #print("videos", videos)
                        #print("persons", persons)
                        #print("frames", frames)
                        #print("data", data)
    
                        
                        # The below code was used since labels were duplicated
                        # index = torch.tensor([0]).to(device)
                        # labels = labels.index_select(1, index)
                        # labels = torch.squeeze(labels)
                        
                        #print(f"labels shape: {labels.shape}")
                        #print("labels", labels)
              
                        optim.zero_grad()
              
                        #if torch.cuda.is_available():
                        #    print('set cross entropy loss to cuda device')
                        #    cross_entropy_loss = nn.CrossEntropyLoss().to(device)     
                        #else:
                        #    cross_entropy_loss = nn.CrossEntropyLoss()
                            
                        #print('cross_entropy_loss.is_cuda', cross_entropy_loss.is_cuda)
                        
                        cross_entropy_loss = nn.CrossEntropyLoss()
                            
                        loss = cross_entropy_loss(output, labels)
                        loss.backward()
                        optim.step()
    
                        #print('labels.size(0)',labels.size(0))
                        train_loss += loss.item() * labels.size(0)
                            
    
    
                    # Evaluate model on validation set
                    the_current_loss, all_outputs, all_labels, all_videos, all_persons = evaluation(model, val_dataloader)
                    all_log_likelihoods = F.log_softmax(all_outputs, dim=1) #nn.CrossEntropyLoss also uses the log_softmax
                    # the class with the highest log-likelihood is what we choose as prediction
                    _, all_predictions = torch.max(all_log_likelihoods, dim=1)          
                    total = all_labels.size(0)
                    correct = (all_predictions == all_labels).sum().item()
                    curr_lr = optim.param_groups[0]['lr']
    
                    #print epoch performance
                    logger.info(f'Fold {fold}, \
                            Epoch {epoch}, \
                            LR:{curr_lr}, \
                            Training Loss: {train_loss/len(train_dataloader):.5f}, \
                            Validation Loss:{the_current_loss:.5f}, \
                            Validation Accuracy: {(correct / total):.4f}, \
                            Time: {((time.time() - temp)/60):.5f} min')
                    
                    #Write epoch performance to file
                    csv_writer_train.writerow([fold, epoch, curr_lr, train_loss/len(train_dataloader), the_current_loss, (correct / total), (time.time() - temp)/60])
                    
                    # Early stopping
    
                    if the_current_loss < min_loss:
                        logger.info('Loss decreased, trigger times: 0')
                        trigger_times = 0
                        min_loss = the_current_loss
                    else:
                        trigger_times += 1
                        logger.info('trigger times: %d', trigger_times)
          
                    if trigger_times > patience or epoch==epochs:
                        logger.info('\nStopping after epoch %d', epoch)
                        
                        temp = time.time()
                        
                        #Save trained model
                        if dataset == "HR-Crime":   
                            PATH = "/data/s3447707/MasterThesis/trained_models/" + model_name + "_fold_" + str(fold) + ".pt"
                        elif dataset == "UTK":
                            PATH = "/data/s3447707/MasterThesis/UTK_trained_models/" + model_name + "_fold_" + str(fold) + ".pt"
                        elif "NTU" in dataset:
                            PATH = "/home/s2435462/HRC/trained_models/" + model_name + "_fold_" + str(fold) + ".pt"
                            
                        #Save trained model
                        torch.save(model, PATH)
                        
                        logger.info("Trained model saved to {}".format(PATH))
    
                        # Evaluate model on test set after training
                        #print('Start evaluating model on at', time.time())
                        # test_dataloader = torch.utils.data.DataLoader([ [traj_categories_test[i], traj_videos_test[i], traj_persons_test[i], traj_frames_test[i], X_test[i] ] for i in range(len(traj_ids_test))], shuffle=True, batch_size=100) 
                        test_dataloader = torch.utils.data.DataLoader(test, batch_size=batch_size, shuffle=True)
                        _, all_outputs, all_labels, all_videos, all_persons = evaluation(model, test_dataloader)
                        all_log_likelihoods = F.log_softmax(all_outputs, dim=1) #nn.CrossEntropyLoss also uses the log_softmax
                        # the class with the highest log-likelihood is what we choose as prediction
                        _, all_predictions = torch.max(all_log_likelihoods, dim=1)
                                                
                        # prepare to count predictions for each class
                        correct_pred = {classname: 0 for classname in all_categories}
                        total_pred = {classname: 0 for classname in all_categories}
                        
    
                          
                        # collect the correct predictions for each class
                        for label, video, person, prediction, log_likelihoods, logits in zip(all_labels, all_videos, all_persons, all_predictions, all_log_likelihoods, all_outputs):
                            
                            csv_writer_test.writerow([fold, label.item(),  video, person, prediction.item(), log_likelihoods.tolist(), logits.tolist()])
                                
                            if label == prediction:
                                correct_pred[all_categories[label]] += 1
                                
                            total_pred[all_categories[label]] += 1
                    
                        total = all_labels.size(0)
                        correct = (all_predictions == all_labels).sum().item()
                
                        logger.info('Accuracy of the network on entire test set: %.2f %% Time: %.5f min' % ( 100 * correct / total, (time.time() - temp)/60 ))
                        
                        # print accuracy for each class
                        for classname, correct_count in correct_pred.items():
                            accuracy = 100 * float(correct_count) / (total_pred[classname] + 0.0000001)
                            logger.info("Accuracy for class {:5s} is: {:.2f} %".format(classname,
                                                                          accuracy))
                    
                        break
    
                    scheduler.step(the_current_loss)
                    
    
                    #print('param groups:', [group['lr'] for group in optim.param_groups])  
                    
                    temp = time.time()
                
                '''
                if dataset == "HR-Crime":   
                    PATH = "/data/s3447707/MasterThesis/trained_models/" + model_name + "_fold_" + str(fold) + ".pt"
                elif dataset == "UTK":
                    PATH = "/data/s3447707/MasterThesis/UTK_trained_models/" + model_name + "_fold_" + str(fold) + ".pt"
                    
                #Save trained model
                torch.save(model, PATH)
                
                print("Trained model saved to {}".format(PATH))
                '''
    
    
    logger.info("Training results saved to {}".format(file_name_train))
    logger.info("Testing results saved to {}".format(file_name_test))


def evaluation(model, data_loader):
    # Settings
    model.eval()
    loss_total = 0
    
    all_outputs = torch.tensor([]).to(device)
    all_labels = torch.LongTensor([]).to(device)
    all_videos = []
    all_persons = []
    # all_videos = torch.LongTensor([]).to(device)
    # all_persons = torch.LongTensor([]).to(device)

    # Test validation data
    with torch.no_grad():
        for batch in data_loader:
            # labels, videos, persons, frames, data = batch
            ids, videos, persons, frames, data, labels = batch
            
            labels = labels.to(device)
            videos = videos#.to(device)
            persons = persons#.to(device)
            frames = frames.to(device)
            data = data.to(device)

            # index = torch.tensor([0]).to(device)
            # labels = labels.index_select(1, index)
            # labels = torch.squeeze(labels)

            #print('videos',videos)
            # videos = videos.index_select(1, index)
            # videos = torch.squeeze(videos)
            # persons = persons.index_select(1, index)
            # persons = torch.squeeze(persons)

            #print('labels length:', len(labels))
            #print('videos:', videos)
            
            outputs = model(data)
            #print('outputs shape:', outputs.shape)
            #print('outputs sum:', torch.sum(outputs, 0))
            #print('\n outputs:',outputs)
            
            #softmax_output = F.softmax(outputs, 1)
            #print('softmax_output sum:', torch.max(softmax_output, 1))
            #print('\nsoftmax_output:',softmax_output)

            #log_softmax_output = F.log_softmax(outputs, 1)
            #print('log_softmax_output sum:', torch.max(log_softmax_output, 1))
            #print('\nlog_softmax_output:', log_softmax_output)

            cross_entropy_loss = nn.CrossEntropyLoss()
            loss = cross_entropy_loss(outputs, labels)  
            loss_total += loss.item()
            
            all_outputs = torch.cat((all_outputs, outputs), 0)
            all_labels = torch.cat((all_labels, labels), 0)
            # print(all_labels)
            all_videos.extend(videos)
            all_persons.extend(persons)
            # all_videos = torch.cat((all_videos, videos), 0)
            # all_persons = torch.cat((all_persons, persons), 0)
            
            #print('all_outputs:', all_outputs)

    return loss_total / len(data_loader), all_outputs, all_labels, all_videos, all_persons
    

#train model
train_model(embed_dim=args.embed_dim, epochs=args.epochs)
