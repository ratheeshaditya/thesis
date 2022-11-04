from itertools import count
import os
import pandas as pd
from csv import reader
import numpy as np
import pickle
from trajectory import Trajectory, get_NTU_categories, remove_short_trajectories, split_into_train_and_test
from utils import SetupLogger

# dimension = '2D'
dimension = '3D'

log_folder = '/home/s2435462/HRC/results/NTU_'+dimension+'decomposing'
os.makedirs(log_folder)
logger = SetupLogger('logger', log_dir=log_folder)

path = '/home/s2435462/HRC/NTU/skeleton/trajectory_csv_'+dimension       

def decompose_trajectories_2D(trajectories_path, decompose_path, classes):
  trajectories = {}
  categories = os.listdir(trajectories_path)

  count_t = 0
  for category in categories:
    category_path = os.path.join(trajectories_path, category) # Path for each category
    folder_names = os.listdir(category_path) # List of folders inside the action class directory
      
    for folder_name in folder_names: # Loop through person folders inside action class directory
        logger.info('load trajectories for video: %s', folder_name)
        folder_path = os.path.join(category_path, folder_name) # Path to person folder
        csv_file_names = os.listdir(folder_path)  # CSV files inside the person folder
        for csv_file_name in csv_file_names: # Loop through csv files inside the person folder
            trajectory_file_path = os.path.join(folder_path, csv_file_name) # Path to trajectory CSV file
            # logger.info(trajectory_file_path)
            try:
                trajectory = np.loadtxt(trajectory_file_path, dtype=np.float32, delimiter=',', ndmin=2) # Load csv using loadtxt
                count_t +=1
            except:
                continue
            person = []
            for frame in trajectory:
                x_local = []
                y_local = []

                trajectory_decomposed = []
                trajectory_decomposed.append(frame[0])

                x_values = frame[1::2]
                y_values = frame[2::2]

                x_g = (min(x_values) + max(x_values))/2
                y_g = (min(y_values) + max(y_values))/2

                trajectory_decomposed.append(x_g)
                trajectory_decomposed.append(y_g)

                w = max(x_values) - min(x_values)
                h = max(y_values) - min(y_values)

                for x, y in zip(x_values, y_values):
                    trajectory_decomposed.append((x-x_g)/w)
                    trajectory_decomposed.append((y-y_g)/h)
                
                # print("x_values:", x_values)
                # print("y_values:", y_values)
                # print("x global: ", x_g)
                # print("y global: ", y_g)
                # print("w: ", w)
                # print("y: ", y)
                # print("x_local: ", x_local)
                # print("y_local: ", y_local)
                # print("trajectory_decomposed: ", trajectory_decomposed)
                # print(len(trajectory_decomposed))
                person.append(trajectory_decomposed)
            person = np.array(person)

            if not os.path.exists(os.path.join(decompose_path, category, folder_name)):
                os.makedirs(os.path.join(decompose_path, category, folder_name))

            with open(os.path.join(decompose_path, category, folder_name, csv_file_name), 'a') as fo:
            # For each frame, write as a line to a CSV file
                for frame in person:
                    pd.DataFrame([frame]).to_csv(fo, header=False, index=False)

  logger.info('count = %d', count_t)

def decompose_trajectories_3D(trajectories_path, decompose_path, classes):
  trajectories = {}
  categories = os.listdir(trajectories_path)

  count_t = 0
  for category in categories:
    category_path = os.path.join(trajectories_path, category) # Path for each category
    folder_names = os.listdir(category_path) # List of folders inside the action class directory
      
    for folder_name in folder_names: # Loop through person folders inside action class directory
        logger.info('load trajectories for video: %s', folder_name)
        folder_path = os.path.join(category_path, folder_name) # Path to person folder
        csv_file_names = os.listdir(folder_path)  # CSV files inside the person folder
        for csv_file_name in csv_file_names: # Loop through csv files inside the person folder
            trajectory_file_path = os.path.join(folder_path, csv_file_name) # Path to trajectory CSV file
            # logger.info(trajectory_file_path)
            try:
                trajectory = np.loadtxt(trajectory_file_path, dtype=np.float32, delimiter=',', ndmin=2) # Load csv using loadtxt
                count_t +=1
            except:
                continue
            person = []
            for frame in trajectory:
                trajectory_decomposed = []
                trajectory_decomposed.append(frame[0])

                x_values = frame[1::3]
                y_values = frame[2::3]
                z_values = frame[3::3]

                x_g = (min(x_values) + max(x_values))/2
                y_g = (min(y_values) + max(y_values))/2
                z_g = (min(z_values) + max(z_values))/2

                trajectory_decomposed.append(x_g)
                trajectory_decomposed.append(y_g)
                trajectory_decomposed.append(z_g)

                w = max(x_values) - min(x_values)
                h = max(y_values) - min(y_values)
                d = max(z_values) - min(z_values)

                for x, y, z in zip(x_values, y_values, z_values):
                    trajectory_decomposed.append((x-x_g)/w)
                    trajectory_decomposed.append((y-y_g)/h)
                    trajectory_decomposed.append((z-z_g)/d)
                
                # print("x_values:", x_values)
                # print("y_values:", y_values)
                # print("x global: ", x_g)
                # print("y global: ", y_g)
                # print("w: ", w)
                # print("y: ", y)
                # print("x_local: ", x_local)
                # print("y_local: ", y_local)
                # print("trajectory_decomposed: ", trajectory_decomposed)
                # print(len(trajectory_decomposed))
                person.append(trajectory_decomposed)
            person = np.array(person)

            if not os.path.exists(os.path.join(decompose_path, category, folder_name)):
                os.makedirs(os.path.join(decompose_path, category, folder_name))

            with open(os.path.join(decompose_path, category, folder_name, csv_file_name), 'a') as fo:
            # For each frame, write as a line to a CSV file
                for frame in person:
                    pd.DataFrame([frame]).to_csv(fo, header=False, index=False)

  logger.info('count = %d', count_t)

decompose_path = '/home/s2435462/HRC/NTU/skeleton/decompose_trajectory_csv_'+dimension 

all_categories = get_NTU_categories()

#decompose trajectories
if dimension=='2D':
    decompose_trajectories_2D(path, decompose_path, all_categories)

if dimension=='3D':
    decompose_trajectories_3D(path, decompose_path, all_categories)