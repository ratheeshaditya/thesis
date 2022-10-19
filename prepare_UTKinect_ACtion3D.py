import re
import pandas as pd
import os
import numpy as np
from trajectory import Trajectory, split_into_train_and_test, get_UTK_categories
import pickle

def load_lables(filename):
    f = open(filename,'r')
    lines = f.read()
    lines = lines.replace('  ',' ') #replace double spacing with single spacing
    lines = re.split(': | \n|\n| ', lines) #split lines
    
    #print(lines)
    
    '''
    for line in lines:
      print(line)
    '''
    
    #actions = ['walk', 'sitDown', 'standUp', 'pickUp', 'carry', 'throw', 'push', 
    #           'pull', 'waveHands', 'clapHands']
    
    actions = get_UTK_categories()
    
    action_labels = []
    
    for i in range(20):
      #print('i', i)
      values = []
      for j in range(31*i,31*i+31):
        #print('j', j)
        if lines[j] not in actions:
          values.append(lines[j])
    
      #print(values)
      action_labels.append(values)
    
    f.close()
    
    
    df_action_labels = pd.DataFrame(action_labels, columns=['sequence_id', 
                                                            'walk_start', 'walk_end',
                                                            'sitDown_start', 'sitDown_end',
                                                            'standUp_start', 'standUp_end',
                                                            'pickUp_start', 'pickUp_end',
                                                            'carry_start', 'carry_end',
                                                            'throw_start', 'throw_end',
                                                            'push_start', 'push_end',
                                                            'pull_start', 'pull_end',
                                                            'waveHands_start', 'waveHands_end',
                                                            'clapHands_start', 'clapHands_end'])
                                                            
    return df_action_labels


def update_file(filename, new_filename):
  updated_data = ''

  new_file = open(new_filename, "w")

  # opening the file
  with open(filename, 'r+') as file:
      # read the file content
      file_content = file.readlines()

      # iterate over the content
      for line in file_content:
          # replace triple spacing with double spacing
          line = line.replace('   ','  ')
        
          # removing last double spacing
          updated_line = ';'.join(line.split('  ')[:-1])
          #print(updated_line)

          # appending data to the variable
          updated_data += f'{updated_line}\n'
        
      #print(updated_data)
      new_file.write(updated_data)
      new_file.close()


def load_UTKinect_trajectories(path, actions):
  trajectories = {}

  for index, row in df_action_labels.iterrows():
    sequence_id = row['sequence_id']
    #print(sequence_id)

    trajectories_path = os.path.join(path, 'trajectories')
    updated_trajectories_path = os.path.join(path, 'updated_trajectories')

    csv_file_name = 'joints_'+ sequence_id + '.txt'
    #print('csv_file_name', csv_file_name)
    trajectory_file_path = os.path.join(trajectories_path, csv_file_name)
    updated_trajectory_file_path = os.path.join(updated_trajectories_path, csv_file_name)
    update_file(trajectory_file_path, updated_trajectory_file_path) #only needed once
    trajectory = np.loadtxt(updated_trajectory_file_path, dtype=np.float32, delimiter=';', ndmin=2)

    is_sequence_id = df_action_labels['sequence_id']==sequence_id
    
    for action_index, action in enumerate(actions):
      #print("{}".format(action))
      start = action + "_start"
      end   = action + "_end"

      #print(df_action_labels[[start, end]])

      #df_sequence_action_label = df_action_labels[['walk_start','walk_end']][is_sequence_id]
      df_sequence_action_label = df_action_labels[[start,end]][is_sequence_id]
      start_frame = df_sequence_action_label.values[0,0]
      end_frame = df_sequence_action_label.values[0,1]
      #print(start_frame)

      if start_frame != 'NaN':
        
        #print(trajectory[:,0]==252)
        is_action = (trajectory[:,0]>=int(start_frame)) & (trajectory[:,0]<=int(end_frame))
        action_trajectory = trajectory[is_action]
        #print(action_trajectory)

        trajectory_frames, trajectory_coordinates = action_trajectory[:, 0].astype(np.int32), action_trajectory[:, 1:]

        trajectory_id = action + '_' + sequence_id
        #print(trajectory_id)
 

        trajectories[trajectory_id] = Trajectory(trajectory_id=trajectory_id,
                                                frames=trajectory_frames,
                                                coordinates=trajectory_coordinates,
                                                category = action_index)

        #print(trajectories[trajectory_id])
        #print('frames:', trajectories[trajectory_id].frames)
        #print('category:', trajectories[trajectory_id].category)
        #print('coordinates:', trajectories[trajectory_id].coordinates)

        #print(len(trajectories[trajectory_id].coordinates[0]))

      else:
        print('no trajectorie available for {} action {}'.format(sequence_id, action))

  return trajectories


path = '/data/s3447707/MasterThesis/UTKinect_Action3D/'
filename = os.path.join(path,'action_lables.txt')

print('Load {}'.format(filename))

df_action_labels = load_lables(filename)

print(df_action_labels)

actions = ['walk', 'sitDown', 'standUp', 'pickUp', 'carry',
           'throw', 'push', 'pull', 'waveHands', 'clapHands']

print(actions)

all_utk_trajectories = load_UTKinect_trajectories(path, actions)
#print(all_utk_trajectories)

train_trajectories, test_trajectories = split_into_train_and_test(all_utk_trajectories, train_ratio=0.8, seed=42)
print('\nThere are %d train trajectories and %d test trajectories' % (len(train_trajectories), len(test_trajectories)))

#save trajectories
PIK = "./data/UTK_trajectories.dat"

with open(PIK, "wb") as f:
  pickle.dump(all_utk_trajectories, f)

print('Saved {} trajectories to {}'.format(len(all_utk_trajectories), PIK))

#save trajectories
PIK_1 = "./data/train_UTK_trajectories.dat"
PIK_2 = "./data/test_UTK_trajectories.dat"

with open(PIK_1, "wb") as f:
  pickle.dump(train_trajectories, f)

with open(PIK_2, "wb") as f:
  pickle.dump(test_trajectories, f)
  
print('%d train UTK trajectories saved to %s' % (len(train_trajectories),PIK_1))
print('%d test UTK trajectories saved to %s' % (len(test_trajectories),PIK_2))


