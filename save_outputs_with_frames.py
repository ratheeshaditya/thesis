 #!/bin/env python
 
#usage
#python save_outputs_with_frames.py --filename FINAL_temporal_transformer_embed_dim_256_segment_length_24 --fold_number 2 --embed_dim 256 --model_type temporal --segment_length 24 --debug
#python save_outputs_with_frames.py --filename FINAL_train_spatial_temporal_transformer_model_cross_val_on_gpu_embed_dim_32_reserve_1_day --fold_number 2 --embed_dim 32 --model_type spatial-temporal
 
 #import packages
import numpy as np
import torch
import torch.nn as nn
from functools import partial
import time
import pickle
from torch.utils.data import DataLoader
import sys
import csv
from sklearn.model_selection import KFold
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch._six import inf
import torch.nn.functional as F
import os
from statistics import mean


from trajectory import Trajectory, extract_fixed_sized_segments, split_into_train_and_test, remove_short_trajectories, get_categories, get_UTK_categories
from transformer import TemporalTransformer_4, TemporalTransformer_3, TemporalTransformer_2, BodyPartTransformer, SpatialTemporalTransformer, TemporalTransformer, Block, Attention, Mlp
from transformer_store_attn import TemporalTransformer_store_attn, TemporalTransformer_2_store_attn, TemporalTransformer_3_store_attn, TemporalTransformer_4_store_attn

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--filename", help="filename to store trained model and results")
parser.add_argument("--fold_number", help="best performing fold", type=int)
parser.add_argument("--embed_dim", help="embedding dimension used by the model", type=int)
parser.add_argument("--debug", help="load subset of trajectories in debug mode", action="store_true", default=False)
parser.add_argument("--model_type", help="type of model to train, temporal, temporal_2, temporal_3, temporal_4, spatial-temporal or parts", type=str)
parser.add_argument("--segment_length", help="length of sliding window", default=12, type=int)
parser.add_argument("--dataset", help="dataset used HR-Crime or UTK", default="HR-Crime", type=str)

args = parser.parse_args()


print('Number of arguments given:', len(sys.argv), 'arguments.')
print('Arguments given:', str(sys.argv))

print('parser args:', args)

#sys.exit()

print('cuda available ', torch.cuda.is_available())
device = "cuda:0" if torch.cuda.is_available() else "cpu"
print ('Available devices ', torch.cuda.device_count())
print ('Current cuda device ', torch.cuda.current_device())


#Load test trajectories
dataset = args.dataset
if dataset=="HR-Crime":
    PIK_test = "./data/test_anomaly_trajectories.dat"
    all_categories = get_categories()
elif dataset == "UTK":
    PIK_test = "./data/test_UTK_trajectories.dat"
    all_categories = get_UTK_categories()
else:
    raise Exception('dataset not recognized, must be HR-Crime or UTK')

with open(PIK_test, "rb") as f:
    test_crime_trajectories = pickle.load(f)

print('\nLoaded %d test trajectories' % (len(test_crime_trajectories)))

test_frame_lengths = []

for key in test_crime_trajectories:
    num_of_frames = len(test_crime_trajectories[key])
    
    test_frame_lengths.append(num_of_frames)

#set segment size
segment_length = args.segment_length
test_crime_trajectories = remove_short_trajectories(test_crime_trajectories, input_length=segment_length, input_gap=0, pred_length=0)

#use subset for when debugging to speed things up, comment  out to train on entire training set
if args.debug:
    test_crime_trajectories = {key: value for key, value in test_crime_trajectories.items() if key[-8:] < '005_0005'}
    print('\nin debugging mode: %d test trajectories' % (len(test_crime_trajectories)))    
else:
    print('\nRemoved short trajectories: %d test trajectories left' % (len(test_crime_trajectories)))

print("\ncategories", all_categories)


#time.sleep(30) # Sleep for 30 seconds to generate memory usage in Peregrine

model_name = args.filename + '_fold_' + str(args.fold_number) #e.g. "transformer_model_embed_dim_32_fold_2"

embed_dim = args.embed_dim
fold = args.fold_number

def evaluate_test_trajectories():
    
    print('Start testing')

    #set batch size
    batch_size = 100
    
    #file to save results
    if dataset == "HR-Crime":
        #file_name_train = '/data/s3447707/MasterThesis/training_results/' + model_name + '.csv'
        file_name_test = '/data/s3447707/MasterThesis/testing_results_with_frames/' + model_name + '.csv'
        num_classes = 13
        num_joints = 17
        num_parts = 5
        in_chans = 2
    elif dataset == "UTK":
        #file_name_train = '/data/s3447707/MasterThesis/UTK_training_results/' + model_name + '.csv'
        file_name_test = '/data/s3447707/MasterThesis/UTK_testing_results_with_frames/' + model_name + '.csv'
        num_classes = 10
        num_joints = 20
        in_chans = 3
        

    #traj_ids_train, traj_videos_train, traj_persons_train, traj_frames_train, traj_categories_train, X_train = extract_fixed_sized_segments(dataset, train_crime_trajectories, input_length=segment_length)
    traj_ids_test, traj_videos_test, traj_persons_test, traj_frames_test, traj_categories_test, X_test = extract_fixed_sized_segments(dataset, test_crime_trajectories, input_length=segment_length)
            
    # prepare to write testing results to a file
    with open(file_name_test, 'w') as csv_file_test:
        csv_writer_test = csv.writer(csv_file_test, delimiter=';')
        csv_writer_test.writerow(['fold', 'label', 'video', 'person', 'frames', 'prediction', 'log_likelihoods', 'logits'])
 
        # Start print
        print('--------------------------------')
    
        print('trajectories to test: %s' % len(test_crime_trajectories))
        
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
        
        
        PATH = '/data/s3447707/MasterThesis/trained_models/' + model_name + '.pt'
        model = torch.load(PATH)
        
        NEW_PATH = '/data/s3447707/MasterThesis/saved_state_dict/' + model_name + '.pt'
        if not os.path.isfile(NEW_PATH):
            torch.save(model.state_dict(), NEW_PATH)
            print('\nsave model state dict to', NEW_PATH)
        else:
            print('\nModel %s already exists' % (NEW_PATH))
    
        
        #Load model state dict
        model.load_state_dict(torch.load(NEW_PATH), strict=False)
        model.to(device)
        model.eval()
        
            
        # Evaluate model on test set
        temp = time.time()
        test_dataloader = torch.utils.data.DataLoader([ [traj_categories_test[i], traj_videos_test[i], traj_persons_test[i], traj_frames_test[i], X_test[i] ] for i in range(len(traj_ids_test))], shuffle=True, batch_size=100) 
        _, all_outputs, all_labels, all_videos, all_persons, all_frames = evaluation(model, test_dataloader)
        all_log_likelihoods = F.log_softmax(all_outputs, dim=1) #nn.CrossEntropyLoss also uses the log_softmax
        # the class with the highest log-likelihood is what we choose as prediction
        _, all_predictions = torch.max(all_log_likelihoods, dim=1)
                                
        # prepare to count predictions for each class
        correct_pred = {classname: 0 for classname in all_categories}
        total_pred = {classname: 0 for classname in all_categories}
          
        # collect the correct predictions for each class
        for label, video, person, frames, prediction, log_likelihoods, logits in zip(all_labels, all_videos, all_persons, all_frames, all_predictions, all_log_likelihoods, all_outputs):
            
            csv_writer_test.writerow([fold, label.item(),  video.item(), person.item(), frames.tolist(), prediction.item(), log_likelihoods.tolist(), logits.tolist()])
                
            if label == prediction:
                correct_pred[all_categories[label]] += 1
                
            total_pred[all_categories[label]] += 1
        
        total = all_labels.size(0)
        correct = (all_predictions == all_labels).sum().item()
        
        print('Accuracy of the network on entire test set: %.2f %% Time: %.5f min' % ( 100 * correct / total, (time.time() - temp)/60 ))
        
        # print accuracy for each class
        for classname, correct_count in correct_pred.items():
            accuracy = 100 * float(correct_count) / (total_pred[classname] + 0.0000001)
            print("Accuracy for class {:5s} is: {:.2f} %".format(classname,
                                                          accuracy))
        

    print("Testing results saved to {}".format(file_name_test))


def evaluation(model, data_loader):
    # Settings
    model.eval()
    loss_total = 0
    
    all_outputs = torch.tensor([]).to(device)
    all_labels = torch.LongTensor([]).to(device)
    all_videos = torch.LongTensor([]).to(device)
    all_persons = torch.LongTensor([]).to(device)
    all_frames = torch.LongTensor([]).to(device)

    # Test validation data
    with torch.no_grad():
        for batch in data_loader:
            labels, videos, persons, frames, data = batch
            
            labels = labels.to(device)
            videos = videos.to(device)
            persons = persons.to(device)
            frames = frames.to(device)
            data = data.to(device)
            
            #print('frames', frames)

            index = torch.tensor([0]).to(device)
            labels = labels.index_select(1, index)
            labels = torch.squeeze(labels)

            #print('videos',videos)
            videos = videos.index_select(1, index)
            videos = torch.squeeze(videos)
            persons = persons.index_select(1, index)
            persons = torch.squeeze(persons)

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
            all_videos = torch.cat((all_videos, videos), 0)
            all_persons = torch.cat((all_persons, persons), 0)
            all_frames = torch.cat((all_frames, frames), 0)
            
        #print('all_frames', all_frames)
        #print('all_outputs:', all_outputs)

    return loss_total / len(data_loader), all_outputs, all_labels, all_videos, all_persons, all_frames
    

#train model
evaluate_test_trajectories()
