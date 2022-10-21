 #!/bin/env python
 
from asyncio.log import logger
import pandas as pd
import numpy as np
from sklearn.metrics import balanced_accuracy_score, accuracy_score, precision_score, recall_score, f1_score, top_k_accuracy_score
from prettytable import PrettyTable
import argparse
import sys

from utils import SetupLogger

logger = SetupLogger('logger')

parser = argparse.ArgumentParser()
parser.add_argument("--filename", help="filename of resulst used to calculate model performance")

args = parser.parse_args()

logger.info('Number of arguments: %d', len(sys.argv))
logger.info('Argument List:', str(sys.argv))

def conv_to_float(x):
  return [float(y) for y in x[1:-1].split(',')]

def evaluate_all(df, category, t):
  fold = df['fold']
  y_true = np.array(df['label'])
  y_pred = df['prediction']
  # y_score = np.array(df_results['log_likelihoods'].values.tolist())
  y_score = df_results['log_likelihoods'].apply(conv_to_float).values.tolist()
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
  top_3_accuracy = top_k_accuracy_score(y_true, y_score, k=3)
  top_5_accuracy = top_k_accuracy_score(y_true, y_score, k=5)

  evaluations = [category, '%.4f' % accuracy, '%.4f' % balanced_accuracy, '%.4f' % weighted_precision, '%.4f' % weighted_recall, '%.4f' % weighted_f1, '%.4f' % top_3_accuracy, '%.4f' % top_5_accuracy]

  t.add_row(evaluations)
  return t
 
def evaluate_category(df, category, t):
  y_true = df['label']
  y_pred = df['prediction']

  accuracy = accuracy_score(y_true, y_pred, normalize=True)

  evaluations = [category, '%.4f' % accuracy]

  t.add_row(evaluations)
  return t

#load training results
#file to save results
# file_name = '/home/s2435462/HRC/results/' + args.filename + '.csv'
file_name = '/home/s2435462/HRC/results/NTU_2D/testing/' + args.filename + '.csv'
logger.info('before read_csv')
df_results = pd.read_csv(file_name, delimiter=';')
logger.info('after read_csv')


headers = ['CATEGORY','ACCURACY(M)','ACCURACY(W)','PRECISION(W)','RECALL(W)','F1-SCORE(W)', 'TOP_3_ACC', 'TOP_5_ACC']

# Evaluate model performance on all crime categories
t_all = PrettyTable(headers)
t_all = evaluate_all(df_results, 'ALL', t_all)
logger.info('\n' + str(t_all))


#Evalutate performace per class

# df_abuse = df_results[df_results['label'] == 0]
# df_arrest = df_results[df_results['label'] == 1]
# df_arson = df_results[df_results['label'] == 2]
# df_assault = df_results[df_results['label'] == 3]
# df_burglary = df_results[df_results['label'] == 4]
# df_explosion = df_results[df_results['label'] == 5]
# df_fighting = df_results[df_results['label'] == 6]
# df_roadaccidents = df_results[df_results['label'] == 7]
# df_robbery = df_results[df_results['label'] == 8]
# df_shooting = df_results[df_results['label'] == 9]
# df_shoplifting = df_results[df_results['label'] == 10]
# df_stealing = df_results[df_results['label'] == 11]
# df_vandalism = df_results[df_results['label'] == 12]


# headers = ['CATEGORY','ACCURACY']
# t = PrettyTable(headers)
# t = evaluate_category(df_abuse, 'Abuse', t)
# t = evaluate_category(df_arrest, 'Arrest', t)
# t = evaluate_category(df_arson, 'Arson', t)
# t = evaluate_category(df_assault, 'Assault', t)
# t = evaluate_category(df_burglary, 'Burglary', t)
# t = evaluate_category(df_explosion, 'Explosion', t)
# t = evaluate_category(df_fighting, 'Fighting', t)
# t = evaluate_category(df_roadaccidents, 'Road Accidents', t)
# t = evaluate_category(df_robbery, 'Robbery', t)
# t = evaluate_category(df_shooting, 'Shooting', t)
# t = evaluate_category(df_shoplifting, 'Shoplifting', t)
# t = evaluate_category(df_stealing, 'Stealing', t)
# t = evaluate_category(df_vandalism, 'Vandalism', t)

# print(t)

#write tables to file
file_name = '/home/s2435462/HRC/results/model_performance/' + args.filename + '.txt'
with open(file_name, 'w') as w:
    w.write(str(t_all))
    w.write('\n\n')
    w.write(str(t_all))