import pickle
from trajectory import Trajectory

#Load test trajectories
PIK = "./data/trajectories.dat"

with open(PIK, "rb") as f:
    trajectories = pickle.load(f)

filename = './trajectory_splits/anomaly_train_clustering_gmm'
infile = open(filename,'rb')
train_trajectories_gmm = pickle.load(infile)
infile.close()

filename = './trajectory_splits/anomaly_test_clustering_gmm'
infile = open(filename,'rb')
test_trajectories_gmm = pickle.load(infile)
infile.close()

train_crime_trajectories = trajectories.copy()

test_crime_trajectories = trajectories.copy()

for key in trajectories:
  if key not in train_trajectories_gmm:
    #print('key:',key)
    train_crime_trajectories.pop(key)
    

for key in trajectories:
  if key not in test_trajectories_gmm:
    #print('key:',key)
    test_crime_trajectories.pop(key)
    
len(train_crime_trajectories)

len(test_crime_trajectories)


#save trajectories
PIK_1 = "./data/train_anomaly_trajectories.dat"
PIK_2 = "./data/test_anomaly_trajectories.dat"

with open(PIK_1, "wb") as f:
  pickle.dump(train_crime_trajectories, f)

with open(PIK_2, "wb") as f:
  pickle.dump(test_crime_trajectories, f)
  
print('%d train anomaly trajectories saved to %s' % (len(train_crime_trajectories),PIK_1))
print('%d test anomaly trajectories saved to %s' % (len(test_crime_trajectories),PIK_2))