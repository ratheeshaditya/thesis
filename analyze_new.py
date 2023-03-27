import os
import pickle


"""
1) Prepare a generic input structure
2) Remove
"""


if __name__ =="__main__":
    path = "../../../../../deepstore/datasets/dmb/MachineLearning/HRC/data/HRC/trajectories_train_HRC_2D.dat"
    # load = open(path,"rb")
    # data = pickle.load(path)
    with open(path, "rb") as f:
     train_crime_trajectories = pickle.load(f)
    # print(train_crime_trajectories)