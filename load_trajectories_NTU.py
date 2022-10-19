 #!/bin/env python
 
 #import packages
from itertools import count
import os
from csv import reader
import numpy as np
import pickle
from trajectory import Trajectory, get_NTU_categories

# dimension = '2D'
dimension = '3D'

path = '/home/s2435462/HRC/NTU/skeleton/trajectory_csv_'+dimension

# class Trajectory:
#     def __init__(self, trajectory_id, frames, coordinates, category, dimension):
#         self.trajectory_id = trajectory_id
#         self.person_id = trajectory_id.split('_')[2][0] # Saves the person id in each video
#         self.frames = frames
#         self.coordinates = coordinates
#         #self.is_global = False
#         self.category = category #crime category: Abuse etc. 
#         self.dimension = 2 if dimension=='2D' else 3

#     def __len__(self):
#         return len(self.frames)

#     def is_short(self, input_length, input_gap, pred_length=0):
#         min_trajectory_length = input_length + input_gap * (input_length - 1) + pred_length

#         return len(self) < min_trajectory_length
        

def load_hr_crime_trajectories(trajectories_path, classes):
  trajectories = {}
  categories = os.listdir(trajectories_path)

  count_t = 0
  for category in categories:
      category_path = os.path.join(trajectories_path, category) # Path for each category
      folder_names = os.listdir(category_path) # List of folders inside the action class directory
      #print('category',category)
      
      if "Normal" not in category:
        for folder_name in folder_names: # Loop through person folders inside action class directory
            #if category == 'Arrest':
            print('load trajectories for video',folder_name)
            folder_path = os.path.join(category_path, folder_name) # Path to person folder
            csv_file_names = os.listdir(folder_path)  # CSV files inside the person folder
            #print('csv_file_names', csv_file_names)
            for csv_file_name in csv_file_names: # Loop through csv files inside the person folder
                trajectory_file_path = os.path.join(folder_path, csv_file_name) # Path to trajectory CSV file
                print(trajectory_file_path)
                try:
                  trajectory = np.loadtxt(trajectory_file_path, dtype=np.float32, delimiter=',', ndmin=2) # Load csv using loadtxt
                  count_t +=1
                except:
                  continue
                trajectory_frames, trajectory_coordinates = trajectory[:, 0].astype(np.int32), trajectory[:, 1:]
                # person_id = csv_file_name.split('.')[0]
                person_id = csv_file_name.split('S')[0]
                # trajectory_id = folder_name + '_' + person_id
                trajectory_id = category + '_' + folder_name + '_' + csv_file_name
                #if "Normal" in category:
                  #print('this is a Normal category')
                  #category_index = classes.index("Normal")
                #else:
                category_index = classes.index('A'+category[1:].lstrip('0'))
                
                #print('category_index',category_index)
                trajectories[trajectory_id] = Trajectory(trajectory_id=trajectory_id,
                                                        frames=trajectory_frames,
                                                        coordinates=trajectory_coordinates,
                                                        category = category_index,
                                                        dimension=dimension)

  print('count = ', count_t)
  return trajectories

all_categories = get_NTU_categories()
print("\ncategories", all_categories)

#print('all_categories.index("Abuse")',all_categories.index("Abuse"))

#load trajectories
trajectories = load_hr_crime_trajectories(path, all_categories)
print('\nLoaded %d trajectories.' % len(trajectories))


#save trajectories
PIK = "trajectories_NTU_"+dimension+".dat"

with open(PIK, "wb") as f:
  pickle.dump(trajectories, f)
