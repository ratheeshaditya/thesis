 #!/bin/env python
 
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
from multiprocessing import Pool
from functools import partial
import psutil
import os




from trajectory import Trajectory, extract_fixed_sized_segments, split_into_train_and_test, remove_short_trajectories, get_categories
from transformer import SpatialTemporalTransformer, TemporalTransformer, Block, Attention, Mlp

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--filename", help="filename to store trained model and results")
parser.add_argument("--embed_dim", help="embedding dimension used by the model", type=int)
parser.add_argument("--debug", help="load subset of trajectories in debug mode", action="store_true", default=False)
#parser.add_argument("--epochs", help="maximum number of epochs during training", default=10000, type=int)
parser.add_argument("--epochs", help="maximum number of epochs during training", default=150, type=int)
parser.add_argument("--patience", help="patience before early stopping is enabled", default=5, type=int)
parser.add_argument("--k_fold", help="number of folds used for corss-validation", default=3, type=int)
parser.add_argument("--lr", help="starting learning rate for adaptive learning", default=0.001, type=float)
parser.add_argument("--lr_patience", help="patience before learning rate is decreased", default=3, type=int)
parser.add_argument("--model_type", help="type of model to train, temporal or spatial-temporal", type=str)

args = parser.parse_args()

print('Number of arguments given:', len(sys.argv), 'arguments.')
print('Arguments given:', str(sys.argv))

print('parser args:', args)

#sys.exit()

all_categories = get_categories()
print("\ncategories", all_categories)

num_cpus = os.environ['SLURM_JOB_CPUS_PER_NODE']
#num_cpus = psutil.cpu_count(logical=False)
print('num_cpus', num_cpus)

#Load trajectories
PIK_train = "./data/train_anomaly_trajectories.dat"
PIK_test = "./data/test_anomaly_trajectories.dat"

device = torch.device("cuda:0") # run on GPU
print ('Available devices ', torch.cuda.device_count())
print ('Current cuda device ', torch.cuda.current_device())

with open(PIK_train, "rb") as f:
    train_crime_trajectories = pickle.load(f)

with open(PIK_test, "rb") as f:
    test_crime_trajectories = pickle.load(f)

#use subset for when debugging to speed things up, comment  out to train on entire training set
if args.debug:
    #train_crime_trajectories = {key: value for key, value in train_crime_trajectories.items() if key < 'Arrest'}
    train_crime_trajectories = {key: value for key, value in train_crime_trajectories.items() if key[-8:] < '001_0005'}
    test_crime_trajectories = {key: value for key, value in test_crime_trajectories.items() if key[-8:] < '001_0005'}
    print('\nin debugging mode: %d train trajectories and %d test trajectories' % (len(train_crime_trajectories), len(test_crime_trajectories)))    
else:
    print('\nLoaded %d train trajectories and %d test trajectories' % (len(train_crime_trajectories), len(test_crime_trajectories)))
   
#extract fixed sized segments using sliding window
segment_length = 12

#time.sleep(30) # Sleep for 30 seconds to generate memory usage in Peregrine

model_name = args.filename
#model_name = "transformer_model_embed_dim_32"
#PATH = "./trained_models/" + model_name + ".pt"

embed_dim = args.embed_dim


# worker function for multiprocessing
def mp_worker(fold_split, 
              traj_videos_train, traj_persons_train, traj_frames_train, traj_categories_train, X_train, 
              traj_ids_test,
              traj_videos_test, traj_persons_test, traj_frames_test, traj_categories_test, X_test,
              embed_dim, epochs):
    fold, (train_ids, val_ids) = fold_split
    print('fold: %s, train: %s, test: %s' % (fold, len(train_ids), len(val_ids)))
    train_dataloader = torch.utils.data.DataLoader([ [traj_categories_train[i], traj_videos_train[i], traj_persons_train[i], traj_frames_train[i], X_train[i]] for i in train_ids], shuffle=True, batch_size=100)
    val_dataloader = torch.utils.data.DataLoader([ [traj_categories_train[i], traj_videos_train[i], traj_persons_train[i], traj_frames_train[i], X_train[i]] for i in val_ids], shuffle=True, batch_size=100)
            
    #intialize model
    if args.model_type == 'temporal':
        model = TemporalTransformer(embed_dim=embed_dim, num_classes=13, mlp_ratio=2., qkv_bias=True, qk_scale=None, dropout=0.1)
    elif args.model_type == 'spatial-temporal':
        model = SpatialTemporalTransformer(embed_dim_ratio=embed_dim, num_classes=13, mlp_ratio=2., qkv_bias=True, qk_scale=None, dropout=0.1)
    else:
        raise Exception('model_type is missing, must be temporal or spatial-temporal')

    # Initialize parameters with Glorot / fan_avg.
    # This code is very important! It initialises the parameters with a
    # range of values that stops the signal fading or getting too big.
    for p in model.parameters():
        if p.dim() > 1:
            #print('parameter:',p)
            nn.init.xavier_uniform_(p)
      
    optim = torch.optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.98), eps=1e-9)

    lr_patience = args.lr_patience
    print('lr_patience', lr_patience)
    scheduler = ReduceLROnPlateau(optim, patience = lr_patience, verbose=True) #patience < early stopping patience
      
    # Early stopping parameters
    min_loss = inf
    patience =  args.patience
    print('Early stopping patience', patience)
    trigger_times = 0
      
    start = time.time()
    temp = start

    #file to save results
    file_name_train = '/data/s3447707/MasterThesis/training_results/training_results_' + model_name + '_fold_' + str(fold) + '.csv'
    file_name_test = '/data/s3447707/MasterThesis/testing_results/testing_results_'  + model_name + '_fold_' + str(fold) + '.csv'

    with open(file_name_train, 'w') as csv_file_train:
        csv_writer_train = csv.writer(csv_file_train, delimiter=';')
        csv_writer_train.writerow(['fold', 'epoch', 'LR', 'Training Loss', 'Validation Loss', 'Validation Accuracy', 'Time'])

        # prepare to write testing results to a file
        with open(file_name_test, 'w') as csv_file_test:
          csv_writer_test = csv.writer(csv_file_test, delimiter=';')
          csv_writer_test.writerow(['fold', 'label',  'video', 'person', 'prediction', 'log_likelihoods', 'logits'])

          for epoch in range(1, epochs+1):

                  train_loss = 0.0
                  
                  model.train()
            
                  for iter, batch in enumerate(train_dataloader, 1):
                  
                      labels, videos, persons, frames, data = batch

                      output = model(data)
                  
                      #print(f"output shape: {output.shape}")
                      #print("output", output)

                      
                      index = torch.tensor([0])
                      labels = labels.index_select(1, index)
                      labels = torch.squeeze(labels)
                      #print(f"labels shape: {labels.shape}")
                      #print("labels", labels)
            
                      optim.zero_grad()
            
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
                  print(f'Fold {fold}, \
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
                      print('Loss decreased, trigger times: 0')
                      trigger_times = 0
                      min_loss = the_current_loss

                  else:
          
                      trigger_times += 1
                      print('trigger times:', trigger_times)

                      
                  if trigger_times > patience or epoch==epochs:
                      print('\nStopping training on fold %d after epoch %d' % (fold,epoch))

                      # Evaluate model on test set after training
                      test_dataloader = torch.utils.data.DataLoader([ [traj_categories_test[i], traj_videos_test[i], traj_persons_test[i], traj_frames_test[i], X_test[i] ] for i in range(len(traj_ids_test))], shuffle=True, batch_size=100) 
                      _, all_outputs, all_labels, all_videos, all_persons = evaluation(model, test_dataloader)
                      all_log_likelihoods = F.log_softmax(all_outputs, dim=1) #nn.CrossEntropyLoss also uses the log_softmax
                      # the class with the highest log-likelihood is what we choose as prediction
                      _, all_predictions = torch.max(all_log_likelihoods, dim=1)

                      # prepare to count predictions for each class
                      correct_pred = {classname: 0 for classname in all_categories}
                      total_pred = {classname: 0 for classname in all_categories}

                      # collect the correct predictions for each class
                      for label, video, person, prediction, log_likelihoods, logits in zip(all_labels, all_videos, all_persons, all_predictions, all_log_likelihoods, all_outputs):

                          csv_writer_test.writerow([fold, label.item(),  video.item(), person.item(), prediction.item(), log_likelihoods.tolist(), logits.tolist()])

                          if label == prediction:
                              correct_pred[all_categories[label]] += 1

                          total_pred[all_categories[label]] += 1

                      total = all_labels.size(0)
                      correct = (all_predictions == all_labels).sum().item()
              
                      print('Accuracy of the network trained on fold %d on the entire test set: %.2f %%' % (fold, 100 * correct / total))

                      # print accuracy for each class
                      for classname, correct_count in correct_pred.items():
                          accuracy = 100 * float(correct_count) / (total_pred[classname] + 0.0000001)
                          print("Accuracy of fold %d for class %s is: %.2f %%" % (fold, classname, accuracy))
                    
                      break

                  scheduler.step(the_current_loss)          

                  #print('param groups:', [group['lr'] for group in optim.param_groups])  

                  temp = time.time()    

    
    print("Training results saved to {}".format(file_name_train))
    print("Testing results saved to {}".format(file_name_test))

    return fold, model
    

def train_model_mp(embed_dim, epochs):

    #set batch size
    batch_size = 100
    
    # prepare cross validation
    n = args.k_fold
    kf = KFold(n_splits=n, random_state=42, shuffle=True)
    

    traj_ids_train, traj_videos_train, traj_persons_train, traj_frames_train, traj_categories_train, X_train = extract_fixed_sized_segments(train_crime_trajectories, input_length=12)
    traj_ids_test, traj_videos_test, traj_persons_test, traj_frames_test, traj_categories_test, X_test = extract_fixed_sized_segments(test_crime_trajectories, input_length=12)

    # Request cores for multiprocessing
    #p = Pool(processes=n)
    p = Pool(processes=int(num_cpus))
    start = time.time()
    mp_helper = partial(mp_worker, 
                        traj_videos_train=traj_videos_train, 
                        traj_persons_train=traj_persons_train, 
                        traj_frames_train=traj_frames_train, 
                        traj_categories_train=traj_categories_train, 
                        X_train=X_train,
                        traj_ids_test=traj_ids_test,
                        traj_videos_test=traj_videos_test, 
                        traj_persons_test=traj_persons_test, 
                        traj_frames_test=traj_frames_test, 
                        traj_categories_test=traj_categories_test, 
                        X_test=X_test, 
                        embed_dim=embed_dim,
                        epochs=epochs)


    # Start print
    print('-------------------------------------------')   
    print('trajectories segments to train: %s' % len(traj_ids_train))  
    print('trajectories segments to test: %s' % len(traj_ids_test))  

    # K-fold Cross Validation model training and evaluation
    for fold, model in p.imap(mp_helper, enumerate(kf.split(traj_ids_train), 1)):
        #model_name = "transformer_model"
        PATH = "/data/s3447707/MasterThesis/trained_models/" + model_name + "_fold_" + str(fold) + ".pt"
        #Save trained model
        torch.save(model, PATH)  
        print("Trained model of fold %s saved to %s" % (fold,PATH))


def evaluation(model, data_loader):
    # Settings
    model.eval()
    loss_total = 0
    
    all_outputs = torch.tensor([])
    all_labels = torch.LongTensor([])
    all_videos = torch.LongTensor([])
    all_persons = torch.LongTensor([])

    # Test validation data
    with torch.no_grad():
        for batch in data_loader:
            labels, videos, persons, frames, data = batch

            index = torch.tensor([0])
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
            
            #print('all_outputs:', all_outputs)

    return loss_total / len(data_loader), all_outputs, all_labels, all_videos, all_persons

'''
def train_model(embed_dim, epochs):
    
    print('call train_model at', time.time())

    #set batch size
    batch_size = 100
    
    # prepare cross validation
    n = args.k_fold
    kf = KFold(n_splits=n, random_state=42, shuffle=True)
    
    #file to save results
    file_name_train = '/data/s3447707/MasterThesis/training_results/' + model_name + '.csv'
    file_name_test = '/data/s3447707/MasterThesis/testing_results/' + model_name + '.csv'

    traj_ids_train, traj_videos_train, traj_persons_train, traj_frames_train, traj_categories_train, X_train = extract_fixed_sized_segments(train_crime_trajectories, input_length=12)
    traj_ids_test, traj_videos_test, traj_persons_test, traj_frames_test, traj_categories_test, X_test = extract_fixed_sized_segments(test_crime_trajectories, input_length=12)
            
    with open(file_name_train, 'w') as csv_file_train:
        csv_writer_train = csv.writer(csv_file_train, delimiter=';')
        csv_writer_train.writerow(['fold', 'epoch', 'LR', 'Training Loss', 'Validation Loss', 'Validation Accuracy', 'Time'])
        # prepare to write testing results to a file
        with open(file_name_test, 'w') as csv_file_test:
            csv_writer_test = csv.writer(csv_file_test, delimiter=';')
            csv_writer_test.writerow(['fold', 'label', 'video', 'person', 'prediction', 'log_likelihoods', 'logits'])
 
            # Start print
            print('--------------------------------')
    
            print('trajectories to train: %s' % len(train_crime_trajectories))
            
            # K-fold Cross Validation model evaluation
            for fold, (train_ids, val_ids) in enumerate(kf.split(traj_ids_train), 1):
                print('\nfold: %s, train: %s, test: %s' % (fold, len(train_ids), len(val_ids)))
    
                train_dataloader = torch.utils.data.DataLoader([ [traj_categories_train[i], traj_videos_train[i], traj_persons_train[i], traj_frames_train[i], X_train[i]] for i in train_ids], shuffle=True, batch_size=100)
                val_dataloader = torch.utils.data.DataLoader([ [traj_categories_train[i], traj_videos_train[i], traj_persons_train[i], traj_frames_train[i], X_train[i]] for i in val_ids], shuffle=True, batch_size=100)
                  
                #intialize model
                if args.model_type == 'temporal':
                    model = TemporalTransformer(embed_dim=embed_dim, num_classes=13, mlp_ratio=2., qkv_bias=True, qk_scale=None, dropout=0.1)
                elif args.model_type == 'spatial-temporal':
                    model = SpatialTemporalTransformer(embed_dim_ratio=embed_dim, num_classes=13, mlp_ratio=2., qkv_bias=True, qk_scale=None, dropout=0.1)
                else:
                    raise Exception('model_type is missing, must be temporal or spatial-temporal')
                
                # Initialize parameters with Glorot / fan_avg.
                # This code is very important! It initialises the parameters with a
                # range of values that stops the signal fading or getting too big.
                for p in model.parameters():
                    if p.dim() > 1:
                        #print('parameter:',p)
                        nn.init.xavier_uniform_(p)
                
                # Define optimizer
                optim = torch.optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.98), eps=1e-9)
                
                # Define scheduler for adapaptive learning
                lr_patience = args.lr_patience
                scheduler = ReduceLROnPlateau(optim, patience = lr_patience, verbose=True) #learning rate patience < early stopping patience
                  
                # Early stopping parameters
                min_loss = inf
                patience =  args.patience
                print('Early stopping patience', patience)
                trigger_times = 0
                  
                start = time.time()
                temp = start
                
                #print('start looping over epochs at', time.time())
                
                for epoch in range(1, epochs+1):
    
                    train_loss = 0.0
                    
                    model.train()
               
                    for iter, batch in enumerate(train_dataloader, 1):
                    
                        labels, videos, persons, frames, data = batch
    
                        output = model(data)
                     
                        #print(f"output shape: {output.shape}")
                        #print("output", output)
                        
                        index = torch.tensor([0])
                        labels = labels.index_select(1, index)
                        labels = torch.squeeze(labels)
                        #print(f"labels shape: {labels.shape}")
                        #print("labels", labels)
              
                        optim.zero_grad()
              
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
                    print(f'Fold {fold}, \
        Epoch {epoch}, \
        LR:{curr_lr}, \
        Training Loss: {train_loss/len(train_dataloader):.5f}, \
        Validation Loss:{the_current_loss:.5f}, \
        Validation Accuracy: {(correct / total):.4f}, \
        Time: {((time.time() - temp)/60):.5f} min')
                    
                    #Write epoch performance to file
                    csv_writer_train.writerow([fold, epoch, train_loss/len(train_dataloader), the_current_loss, (correct / total), (time.time() - temp)/60])
                    
                    # Early stopping
    
                    if the_current_loss < min_loss:
                        print('Loss decreased, trigger times: 0')
                        trigger_times = 0
                        min_loss = the_current_loss
    
                    else:
            
                        trigger_times += 1
                        print('trigger times:', trigger_times)
                        
                    if trigger_times > patience or epoch==epochs:
                        print('\nStopping after epoch %d' % (epoch))
                        
                        temp = time.time()
    
                        # Evaluate model on test set after training
                        #print('Start evaluating model on at', time.time())
                        test_dataloader = torch.utils.data.DataLoader([ [traj_categories_test[i], traj_videos_test[i], traj_persons_test[i], traj_frames_test[i], X_test[i] ] for i in range(len(traj_ids_test))], shuffle=True, batch_size=100) 
                        _, all_outputs, all_labels, all_videos, all_persons = evaluation(model, test_dataloader)
                        all_log_likelihoods = F.log_softmax(all_outputs, dim=1) #nn.CrossEntropyLoss also uses the log_softmax
                        # the class with the highest log-likelihood is what we choose as prediction
                        _, all_predictions = torch.max(all_log_likelihoods, dim=1)
                                                
                        # prepare to count predictions for each class
                        correct_pred = {classname: 0 for classname in all_categories}
                        total_pred = {classname: 0 for classname in all_categories}
                        
    
                          
                        # collect the correct predictions for each class
                        for label, video, person, prediction, log_likelihoods, logits in zip(all_labels, all_videos, all_persons, all_predictions, all_log_likelihoods, all_outputs):
                            
                            csv_writer_test.writerow([fold, label.item(),  video.item(), person.item(), prediction.item(), log_likelihoods.tolist(), logits.tolist()])
                                
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
                    
                        break
    
                    scheduler.step(the_current_loss)
                    
    
                    #print('param groups:', [group['lr'] for group in optim.param_groups])  
                    
                    temp = time.time()
                
                
                    
                PATH = "/data/s3447707/MasterThesis/trained_models/" + model_name + "_fold_" + str(fold) + ".pt"
                #Save trained model
                torch.save(model, PATH)
                
                print("Trained model saved to {}".format(PATH))
    
    
    print("Training results saved to {}".format(file_name_train))
    print("Testing results saved to {}".format(file_name_test))
'''    

#train model
train_model_mp(embed_dim=args.embed_dim, epochs=args.epochs)
