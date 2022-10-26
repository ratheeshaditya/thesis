from statistics import mean
from sklearn.metrics import balanced_accuracy_score, accuracy_score, precision_score, recall_score, f1_score, top_k_accuracy_score
import logging
import sys
import os
import numpy as np

def SetupFolders(training_name, dataset):
  base_folder = os.path.join('/home/s2435462/HRC/results', dataset, training_name)
  model_dir = os.path.join(base_folder, 'models')
  log_dir = os.path.join(base_folder, 'logs')
  results_dir = os.path.join(base_folder, 'results')

  os.makedirs(model_dir)
  os.makedirs(log_dir)
  os.makedirs(results_dir)

  return base_folder, model_dir, log_dir, results_dir

def SetupLogger(name, log_dir=None):
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s %(message)s')

    sh = logging.StreamHandler(stream=sys.stdout)
    sh.setLevel(logging.INFO)
    sh.setFormatter(formatter)
    logger.addHandler(sh)

    if log_dir:
      fh = logging.FileHandler(os.path.join(log_dir, 'logs.log'))
      fh.setLevel(logging.INFO)
      fh.setFormatter(formatter)
      logger.addHandler(fh)

    return logger

def smaller_than_mean(lengths, mean):
    return len([x for x in lengths if x <= mean])

def print_statistics(train_crime_trajectories, test_crime_trajectories, logger):
    train_frame_lengths = []
    test_frame_lengths = []
    for key in train_crime_trajectories:
        #print(key, ' : ', train_crime_trajectories[key])
        num_of_frames = len(train_crime_trajectories[key])
        #print('number of frames:', num_of_frames)
        
        train_frame_lengths.append(num_of_frames)

    for key in test_crime_trajectories:
        num_of_frames = len(test_crime_trajectories[key])
        
        test_frame_lengths.append(num_of_frames)

    logger.info('TRAIN minimum: %d', min(train_frame_lengths))
    logger.info('TRAIN maximum: %d', max(train_frame_lengths))
    logger.info('TRAIN mean: %f', mean(train_frame_lengths))

    logger.info('TEST minimum: %d', min(test_frame_lengths))
    logger.info('TEST maximum: %d', max(test_frame_lengths))
    logger.info('TEST mean: %f', mean(test_frame_lengths))

    logger.info('TRAIN smaller_than_mean: %d', smaller_than_mean(train_frame_lengths, mean(train_frame_lengths)))
    logger.info('TEST smaller_than_mean: %d', smaller_than_mean(test_frame_lengths, mean(test_frame_lengths)))

def evaluate_all(df, category, t):
  fold = df['fold']
  y_true = np.array(df['label'])
  y_pred = df['prediction']
  # y_score = np.array(df_results['log_likelihoods'].values.tolist())
  y_score = df['log_likelihoods'].apply(conv_to_float).values.tolist()
  # y_score = np.array(df_results['log_likelihoods'].str.strip('[]').str.split().tolist(), dtype='float')
  
  # print(type(y_score))
  # print(y_score[:5])
  # print(type(y_true))

  # print(len(y_score[0]))

  accuracy = accuracy_score(y_true, y_pred, normalize=True)
  balanced_accuracy = balanced_accuracy_score(y_true, y_pred)
  weighted_recall = recall_score(y_true, y_pred, average='weighted')
  weighted_precision = precision_score(y_true, y_pred, average='weighted')
  weighted_f1 = f1_score(y_true, y_pred, average='weighted')
  top_3_accuracy = top_k_accuracy_score(y_true, y_score, k=3, labels=np.arange(120))
  top_5_accuracy = top_k_accuracy_score(y_true, y_score, k=5, labels=np.arange(120))

  results = {}
  results['acc'] = accuracy
  results['bal_acc'] = balanced_accuracy
  results['weighted_R'] = weighted_recall
  results['weighted_P'] = weighted_precision
  results['weighted_f1'] = weighted_f1
  results['top_3_acc'] = top_3_accuracy
  results['top_5_acc'] = top_5_accuracy

  evaluations = [category, '%.4f' % accuracy, '%.4f' % balanced_accuracy, '%.4f' % weighted_precision, '%.4f' % weighted_recall, '%.4f' % weighted_f1, '%.4f' % top_3_accuracy, '%.4f' % top_5_accuracy]

  t.add_row(evaluations)
  return results, t
 
def evaluate_category(df, category, t):
  y_true = df['label']
  y_pred = df['prediction']

  accuracy = accuracy_score(y_true, y_pred, normalize=True)

  evaluations = [category, '%.4f' % accuracy]

  t.add_row(evaluations)
  return t

def conv_to_float(x):
  return [float(y) for y in x[1:-1].split(',')]