import numpy as np

# import h5py
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

from torchvision.models.feature_extraction import create_feature_extractor
import torchvision
import os
from einops import rearrange

import torch.multiprocessing as mp




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

    def __len__(self):
        return len(self.ids)

    def extract_frames(self,path,frame_list):

        cap = cv2.VideoCapture(path) 

        middle_frame = frame_list[len(frame_list)//2].item()
        
        
        cap.set(1,middle_frame) 
        ret, frame = cap.read()

        dim = (224,224)

        frame = cv2.resize(frame, dim, interpolation = cv2.INTER_AREA)

        return frame/255.0

    def __getitem__(self, idx):
        data = {}
        data['id'] = self.ids[idx] #contains all the videos
        
        data['videos'] = self.videos[idx]
        
        data['persons'] = self.persons[idx]
        data['frames'] = self.frames[idx]
        # data['categories'] = self.categories[idx]
        # data['coordinates'] = self.coordinates[idx]
        data['path'] = self.path[idx]
        # print(self.path[idx])
        # data['context'] = self.context[idx]
        # print("Extracted frames time ")
        # start = time.time()
        data['extracted_frames'] = self.extract_frames(self.path[idx],self.frames[idx])
        # data['extracted_frames'] = self.context_frames[idx]
        # print(f"Extracted frames : {abs(start-time.time())}")
        return data
        # return self.ids[idx], self.videos[idx], self.persons[idx], self.frames[idx],self.coordinates[idx], self.categories[idx]

    def trajectory_ids(self):
        return self.ids





def collator_for_lists(batch):
    '''
    Reference : https://stackoverflow.com/questions/64883998/pytorch-dataloader-shows-odd-behavior-with-string-dataset
    Reference : https://stackoverflow.com/questions/52818145/why-pytorch-dataloader-behaves-differently-on-numpy-array-and-list
    '''
    # assert all('sentences' in x for x in batch)
    # assert all('label' in x for x in batch)
    # print(batch[0]["extracted_frames"])
    # print(type(batch[0]["extracted_frames"]))
    a = {
        'id': [x['id'] for x in batch],
        'videos': [x['videos'] for x in batch],
        # 'persons': [x['persons'] for x in batch],
        'frames': torch.tensor(np.array([x['frames'] for x in batch])),
        # 'categories': torch.tensor(np.array([x['categories'] for x in batch])),
        # 'coordinates': torch.tensor(np.array([x['coordinates'] for x in batch])),
        # 'extracted_frames': torch.tensor([np.array(i["extracted_frames"]) for i in batch])
        # 'extracted_frames':torch.stack(list(map(lambda x: torch.tensor(x['extracted_frames']), batch))) 
        # 'extracted_frames':torch.stack(list(map(lambda x: x["extracted_frames"], batch))) 
        'extracted_frames':torch.tensor(np.array([x['extracted_frames'] for x in batch])) 
    }
    # logger.info(f"Total time to retrieve : {abs(start-time.time())}")
    return a
    

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') #code for returning device
print(device)


data_to_use = "test"


train_file_path = f"/home/s2765918/code-et/code/data/HRC/trajectories_{data_to_use}_HRC_2D.dat"

with open(train_file_path, "rb") as f:
    train_crime_trajectories = pickle.load(f)
print(f"Size : {len(train_crime_trajectories)}")

print(f"Loaded {data_to_use}  trajectories")

# print("Loaded test trajectories")
# test_crime_trajectories = pickle.load(test_file_path)


train_crime_trajectories = remove_short_trajectories(train_crime_trajectories, input_length=20, input_gap=0, pred_length=0)
# test_crime_trajectories = remove_short_trajectories(test_crime_trajectories, input_length=20, input_gap=0, pred_length=0)


train = TrajectoryDataset_(*extract_fixed_sized_segments("HRC", train_crime_trajectories, input_length=20))
# test = TrajectoryDataset(*extract_fixed_sized_segments(test_crime_trajectories, test_crime_trajectories, input_length=segment_length))

print(f"Len : {len(train)} ")


batch_size = 125
no_workers = 32
pin_memory= True
persistent_workers = True 


print(f"Batch size {batch_size}")

train_dataloader = torch.utils.data.DataLoader(train, batch_size = batch_size, shuffle=False, 
                                               collate_fn=collator_for_lists)

network = getattr(torchvision.models,"vit_b_16")(pretrained=True)
feature_extractor = create_feature_extractor(network, return_nodes=['getitem_5'])
feature_extractor.to(device)
print("Feature extractor loaded to device")

def process_batch(batch):
    # Your batch processing logic goes here
    # This is just an example, you can modify it as per your requirements
    processed_batch = batch + 1  # Add 1 to each element of the batch

    # Add the processed batch to the output queue
    output_queue.put(processed_batch)

    frames = i["extracted_frames"]

    frames = rearrange(frames,"b h w c -> b c h w").to(device,dtype=torch.float)

    features = feature_extractor(frames)

    features_cpu =  features["getitem_5"].cpu().detach().numpy()

    return features_cpu
# features_array = []

def main():
    for batch in data_loader:
        process_batch(batch)

    # Signal the end of processing to the output queue
    output_queue.put(None)

processes = []
for _ in range(num_workers):
    process = mp.Process(target=main)
    process.start()
    processes.append(process)

for process in processes:
    process.join()

processed_batches = []
while True:
    batch = output_queue.get()
    if batch is None:
        break
    processed_batches.append(batch)

# Concatenate the processed batches into a single NumPy array or perform any further processing as needed
processed_data = np.concatenate(processed_batches, axis=0)



print("Writing to file : {}")
np.save(f"/home/s2765918/code-et/code/npy_data/{data_to_use}_all_20_vit16_new.npy",array)
print("Completed. Finshing")


#This is for train
# memmap_array = np.memmap(f"/home/s2765918/code-et/code/npy_data/{data_to_use}_all_20_vit16.npy", dtype='float32', mode='w+', shape=(len(train),768))
# array = np.empty(shape=(len(train),768),dtype=np.float32)


# print("Extracting features..starting")
# j = 0
# for index,i in enumerate(train_dataloader):
#     # print(index)
#     frames = i["extracted_frames"]

#     frames = rearrange(frames,"b h w c -> b c h w").to(device,dtype=torch.float)

#     features = feature_extractor(frames)

#     features_cpu =  features["getitem_5"].cpu().detach().numpy()
#     start_idx = index * batch_size #point towards the next batch

#     end_idx = start_idx + len(features_cpu) #adds the index with the retrieved batch size
#     array[start_idx:end_idx] = features_cpu
#     # print(memmap_array.shape)
#     # features_array.append(features_cpu)
#     # memmap_array.flush()
#     j=j+1
#     if j%100==0:
#         print("Completed 100")
#         break
    
#     # print(j)
#     del frames
#     del features
#     del features_cpu
#     # print(np.stack(features_array).shape)


# del memmap_array
    
    

# features_array = np.concatenate(features_array)
# print(features_array.shape)
