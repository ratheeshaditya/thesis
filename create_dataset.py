import h5py
import pickle
# from vit import VisionTransformer
import torch.nn.functional as F
from torch.utils.data import Dataset
import random
import torch
from matplotlib import pyplot as plt
import cv2
from torch import nn
# from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize
import matplotlib.pyplot as plt
from trajectory import Trajectory, TrajectoryDataset, extract_fixed_sized_segments, split_into_train_and_test, remove_short_trajectories, get_categories, get_UTK_categories, get_NTU_categories,extract_frames
import pickle
import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit
from einops import rearrange, repeat
# from transformer import TubeletTemporalSpatialPart_concat_chan_2_Transformer, \
#                 TubeletTemporalPart_concat_chan_1_Transformer, TubeletTemporalTransformer, \
#                 TubeletTemporalPart_mean_chan_1_Transformer, TubeletTemporalPart_mean_chan_2_Transformer,\
#                  TubeletTemporalPart_concat_chan_2_Transformer, TemporalTransformer_4, \
#                  TemporalTransformer_3, TemporalTransformer_2, BodyPartTransformer, \
#                  SpatialTemporalTransformer, TemporalTransformer, Block, Attention, Mlp, ensemble,vit_model


def collator_for_lists(batch):
    '''
    Reference : https://stackoverflow.com/questions/64883998/pytorch-dataloader-shows-odd-behavior-with-string-dataset
    Reference : https://stackoverflow.com/questions/52818145/why-pytorch-dataloader-behaves-differently-on-numpy-array-and-list
    '''
    # assert all('sentences' in x for x in batch)
    # assert all('label' in x for x in batch)
    
    a = {
        # 'id': [x['id'] for x in batch],
        # 'videos': [x['videos'] for x in batch],
        # 'persons': [x['persons'] for x in batch],
        # 'frames': torch.tensor(np.array([x['frames'] for x in batch])),
        # 'categories': torch.tensor(np.array([x['categories'] for x in batch])),
        # 'coordinates': torch.tensor(np.array([x['coordinates'] for x in batch])),
        # 'extracted_frames': torch.tensor([np.array(i["extracted_frames"]) for i in batch])
        # 'extracted_frames':torch.stack(list(map(lambda x: torch.tensor(x['extracted_frames']), batch))) 
        # 'extracted_frames':torch.stack(list(map(lambda x: x["extracted_frames"], batch))) 
                'extracted_frames': np.array([x['extracted_frames'] for x in batch]),
    }
    # logger.info(f"Total time to retrieve : {abs(start-time.time())}")
    return a
    

import os
def getVideoPath(ids,path="/deepstore/datasets/dmb/MachineLearning/HRC/HRC_files/UCF_Videos"):
    # print(ids)
    getVideoFiles = ids
    # print("there are ids")
    # print(getVideoFiles)
    file = getVideoFiles.split("_")[0]
    
    videoDir = file[:-3] 

    videofile = file +"_x264.mp4"

    video_path = os.path.join(path,videoDir,videofile)

    return video_path

class TrajectoryDataset_(Dataset):
    """
    A dataset to store the trajectories. This should be more efficient than using just arrays.
    Also should be efficient with dataloaders.
    """
    def __init__(self, trajectory_ids, trajectory_videos, trajectory_persons, trajectory_frames, trajectory_categories, X):
        self.ids = trajectory_ids.tolist()
        print("list of ids")
        print(len(self.ids))
        self.videos = trajectory_videos.tolist()
        self.persons = trajectory_persons.tolist()
        self.frames = trajectory_frames
        self.categories = trajectory_categories
        self.coordinates = X
    
        self.path = [getVideoPath(i[0]) for i in self.ids]
        # cold_start = np.stack([extract_frames(self.ids[i],self.frames[i]) for i in range(1000)])
        # f = h5py.File('context_dataset.hdf5','a')
        # start_idx= 1000
        # cold_start = np.stack([extract_frames(self.ids[i],self.frames[i]) for i in range(start_idx)])

        # f.create_dataset("context_data", data=cold_start,maxshape=(None,224,224,3),compression="gzip")
        # print("Creating h5py file with compression on all data")
        # cold_start =[]
        # for i in range(start_idx,len(self.ids)):
        #     cold_start.append(extract_frames(self.ids[i+1],self.frames[i+1]))
        #     cold_start = np.stack(cold_start,axis=0)
        #     if (i + 1) % 1000 == 0 or (i + 1) == len(self.ids):
        #         f["context_data"].resize((f["context_data"].shape[0] + cold_start.shape[0]), axis = 0)
        #         f["context_data"][-cold_start.shape[0]:] = cold_start
        #         cold_start=[]
        #     if i%100000:
        #         print("Completed")
        # f.close()
        # print("Created H5py file")
    def __len__(self):
        return len(self.ids)

    def extract_frames(self,idx,frame_list,path="/deepstore/datasets/dmb/MachineLearning/HRC/HRC_files/UCF_Videos"):
        

        try:
            cap = cv2.VideoCapture(self.path[idx]) 

            middle_frame = frame_list[len(frame_list)//2].item()
            
            cap.set(1,middle_frame) 

            ret, frame = cap.read()
            # dim = (224,224)

            # resized = cv2.resize(frame, dim, interpolation = cv2.INTER_AREA)

            # frame = rearrange(resized,"h w c -> c h w")
            return frame/255.0
        except:
            print(self.path[idx])
            print("Data not found for this file")
        # return torch.from_numpy(resized)/255.0

    def __getitem__(self, idx):
        data = {}
        data['id'] = self.ids[idx] #contains all the videos
        
        data['videos'] = self.videos[idx]
        
        data['persons'] = self.persons[idx]
        data['frames'] = self.frames[idx]
        data['categories'] = self.categories[idx]
        data['coordinates'] = self.coordinates[idx]
        data['path'] = self.path[idx]
        # print(self.path[idx])
        # data['context'] = self.context[idx]
        # print("Extracted frames time ")
        # start = time.time()
        data['extracted_frames'] = self.extract_frames(idx,self.frames[idx])
        # data['extracted_frames'] = self.context_frames[idx]
        # print(f"Extracted frames : {abs(start-time.time())}")
        return data
        # return self.ids[idx], self.videos[idx], self.persons[idx], self.frames[idx],self.coordinates[idx], self.categories[idx]

    def trajectory_ids(self):
        return self.ids

# PIK = "/home/s2765918/code-et/code/data/HRC/trajectories_test_HRC_2D.dat"

# with open(PIK, "rb") as f:
#     trajectories = pickle.load(f)
# print("Loaded train trajectories")



# filepath = '/home/s2765918/code-et/code/hdf5_dataset/context_dataset_20segments_test.hdf5'
# print("Created Hdf5 file")
# train_crime_trajectories = remove_short_trajectories(trajectories, input_length=20, input_gap=0, pred_length=0)
# train = TrajectoryDataset_(*extract_fixed_sized_segments("HRC", train_crime_trajectories, input_length=20))
# dataloader = torch.utils.data.DataLoader(train, batch_size=1000, shuffle=False, collate_fn=collator_for_lists,num_workers=10,pin_memory=True,persistent_workers=True)

# dataset_name = "context_images"

# with h5py.File(filepath, "a") as file:
# # Create the dataset with an initial shape based on the first batch of samples
#     first_batch = next(iter(dataloader))["extracted_frames"]
#     dataset_shape = (len(train),) + first_batch.shape[1:]  # Update shape based on the first batch
#     dataset_dset = file.create_dataset(dataset_name, shape=dataset_shape, dtype=first_batch.dtype,
#                                         chunks=(500,) + first_batch.shape[1:], compression="gzip")



#     # Iterate through the dataloader and append samples to the dataset
#     index = 0
#     print("started")

#     for batch in dataloader:
#         print(batch["extracted_frames"].shape)
#         print("writing")
#         batch_samples = batch["extracted_frames"]  # Assuming samples are stored in "extracted_frames"
#         num_samples = batch_samples.shape[0]
#         dataset_dset[index:index + num_samples] = batch_samples
#         index += num_samples
#     print("completed")
    
    
# Read and verify the shape of the dataset from the HDF5 file
h5 = h5py.File("/home/s2765918/code-et/code/hdf5_dataset/data.h5", "r") 
for i in range(10):
    idx = sorted(random.sample(range(0, 844268), 1000))
    print(h5["images"][idx])

# Check the shape of the dataset
# print(dataset_shape)  # Output: (total_samples, ...)