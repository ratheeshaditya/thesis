 #!/bin/env python
 
 #import packages
import os
from csv import reader
import numpy as np
import pickle
from trajectory import get_categories

path = '/data/s3447707/HR-Crime/Trajectories'

class Trajectory:
    def __init__(self, trajectory_id, frames, coordinates, category):
        self.trajectory_id = trajectory_id
        self.person_id = trajectory_id.split('_')[1]
        self.frames = frames
        self.coordinates = coordinates
        #self.is_global = False
        self.category = category #crime category: Abuse etc. 

    def __len__(self):
        return len(self.frames)

    def is_short(self, input_length, input_gap, pred_length=0):
        min_trajectory_length = input_length + input_gap * (input_length - 1) + pred_length

        return len(self) < min_trajectory_length
        

def load_hr_crime_trajectories(trajectories_path, classes):
  trajectories = {}
  categories = os.listdir(trajectories_path)

  for category in categories:
      category_path = os.path.join(trajectories_path, category)
      folder_names = os.listdir(category_path)
      #print('category',category)
      
      if "Normal" not in category:
        for folder_name in folder_names:
            #if category == 'Arrest':
            print('load trajectories for video',folder_name)
            folder_path = os.path.join(category_path, folder_name)
            csv_file_names = os.listdir(folder_path)
            #print('csv_file_names', csv_file_names)
            for csv_file_name in csv_file_names:
                trajectory_file_path = os.path.join(folder_path, csv_file_name)
                trajectory = np.loadtxt(trajectory_file_path, dtype=np.float32, delimiter=',', ndmin=2)
                trajectory_frames, trajectory_coordinates = trajectory[:, 0].astype(np.int32), trajectory[:, 1:]
                person_id = csv_file_name.split('.')[0]
                trajectory_id = folder_name + '_' + person_id
                #if "Normal" in category:
                  #print('this is a Normal category')
                  #category_index = classes.index("Normal")
                #else:
                category_index = classes.index(category)
                
                #print('category_index',category_index)
                trajectories[trajectory_id] = Trajectory(trajectory_id=trajectory_id,
                                                        frames=trajectory_frames,
                                                        coordinates=trajectory_coordinates,
                                                        category = category_index)

  return trajectories

all_categories = get_categories()
print("\ncategories", all_categories)

#print('all_categories.index("Abuse")',all_categories.index("Abuse"))

#load trajectories
trajectories = load_hr_crime_trajectories(path, all_categories)
print('\nLoaded %d trajectories.' % len(trajectories))

#save trajectories
PIK = "trajectories.dat"

with open(PIK, "wb") as f:
  pickle.dump(trajectories, f)
