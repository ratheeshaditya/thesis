 #!/bin/env python
 
import pickle
from trajectory import Trajectory, remove_short_trajectories, split_into_train_and_test

#load trajectories
PIK = "./data/trajectories.dat"

with open(PIK, "rb") as f:
    trajectories = pickle.load(f)
    
print('\nLoaded %d trajectories.' % len(trajectories))


#remove short trajectories
trajectories = remove_short_trajectories(trajectories, input_length=12, input_gap=0, pred_length=12)

print('\nRemoved short trajectories. Number of trajectories left: %d.' % len(trajectories))

#split trajectories into train and test
trajectories_train, trajectories_test = split_into_train_and_test(trajectories, train_ratio=0.8, seed=42)
print('\n%d train trajectories and %d test trajectories' % (len(trajectories_train), len(trajectories_test)))

#save trajectories for train and test
PIK_train = "./data/trajectories_train.dat"
PIK_test = "./data/trajectories_test.dat"

with open(PIK_train, "wb") as f:
  pickle.dump(trajectories_train, f)
 
with open(PIK_test, "wb") as f:
  pickle.dump(trajectories_test, f)
