from torch.utils.data import Dataset
import os
from tqdm import tqdm
import numpy as np
import pickle

"""
Defining the dataset of HRC and NTU

"""


class hrc_dataset(Dataset):
    def __init__(self,folderpath="../../../../../deepstore/datasets/dmb/MachineLearning/HRC/HRC_files",
                trajectory_path="dataverse_files/trajectories_all",video_path="UCF_Videos",print_stats=True):
        
        self.dataset_name = "hrc"
        self.root_folder = folderpath
        
        #Relative to the folderpath variable
        self.relative_videopath=  video_path
        self.relative_trajectorypath = trajectory_path
        
        #Full path
        self.video_path = os.path.join(self.root_folder,self.relative_videopath)
        self.trajectory_path = os.path.join(self.root_folder,self.relative_trajectorypath)
        
        #Meta details
        self.no_classes = os.listdir(self.trajectory_path)
        self.category = sorted([category for category in os.listdir(self.trajectory_path)])
        self.class_ix = {category:index for index,category in enumerate(self.category)}
        self.ix_class = {index:category for index,category in enumerate(self.category)}

        if print_stats==True:

            print("Dataset statistics")
            print("="*30)
            print(f"Total number of categories : {len(os.listdir(self.trajectory_path))}")
            print(f"All Categories : {os.listdir(self.trajectory_path)}")
            # self.calculate_stats()

    """
    This function creates the structure for the files in this format
    required to run only once.
    {
        "category_name" : 
        "video_file_name" : 
        "frames" : {
                        person_id: {frames} #1 being the frame number, type tensor
                    }
    }
    """

    def create_data(self):
        if not os.path.exists(self.dataset_name+"data"):
            os.mkdir(os.path.join(self.dataset_name+"data"))
        dir_to_create = self.dataset_name+"data"
        
        all_data={"class_ix":self.class_ix,"ix_class":self.ix_class,"all_data":[]}

        for category in self.category:
            trajectory_list = os.path.join(self.trajectory_path,category)
            for trajectory in os.listdir(trajectory_list): #abuse001,abuse002 folders
                temp_data = {"category_name":category,
                            "video_file_name":trajectory,
                            "class_idx":self.class_ix[category]}
                temp_frame = {} #mapping from frame to person
                person_frame = {} #mapping from person to frame
                trajectory_path = os.path.join(trajectory_list,trajectory)
                # print(trajectory_path)
                if trajectory!=".DS_Store":
                    for person_file in os.listdir(trajectory_path): #001.csv,#002.csv
                        if person_file!=".DS_Store":
                            full_personpath = os.path.join(trajectory_path,person_file)
                            
                            data = np.loadtxt(full_personpath, delimiter=',',ndmin=2)
                            data = np.array(sorted(data,key=lambda x:x[0]))
                            only_coords = data[:,1:]
                            only_frames = data[:,0]
                            person_name = person_file.split(".")[0]
                            # print(type(only_coords))
                            for index,(frame,coords) in enumerate(zip(only_frames,only_coords)):
                                
                                #mapping for frame->person
                                if int(frame) not in temp_frame:
                                    temp_frame[int(frame)] = {person_name:coords}
                                else:
                                    temp_frame[int(frame)][person_name]=coords
                                #creating mapping for person -> frame     
                                if person_name not in person_frame:
                                    person_frame[person_name] = [int(frame)]
                                else:
                                    person_frame[person_name].append(int(frame))
                    temp_data["video_frames"] = temp_frame
                    temp_data["person_frame"] = person_frame
                all_data["all_data"].append(temp_data)
                

        filename = f"all_data-{self.dataset_name}.pickle"
        with open(f"{os.path.join(dir_to_create,filename)}", 'wb') as handle:
            pickle.dump(all_data, handle, protocol=pickle.HIGHEST_PROTOCOL)





    # def __getitem__(self,index,category="abuse"):

        


    # def print_stats(self):


    # def __len__(self):
        #pass


    # def getFilePaths(classname="Abuse",index=0):

    #     category = classname
    #     trajectory_path = os.path.join(self.folder_path,self.trajectory_path)

    #     video_path = os.path.join(path_hrc,"UCF_Videos")

    #     trajectory_category = os.path.join(trajectory_path,category)

    #     video_category = os.path.join(video_path,category)
        
    #     all_files_video = sorted(os.listdir(video_category))
    

    #     all_folders_trajectory = sorted(os.listdir(trajectory_category))
        
    #     file_name = all_files_video[index]
        
    #     # print(file_name)
    #     video_full_path = os.path.join(video_category,file_name)
    #     # print(video_full_path)

    #     trajectory_full_path = os.path.join(trajectory_category,file_name.split("_")[0])
    #     # print(trajectory_full_path)
    #     return video_full_path,trajectory_full_path


    # def __getitem__(self):
        #pass


# class ntu_dataset(Dataset):
#     def __init__(self):
#         #pass

#     def __len__(self):
#         #pass
    
#     def __getitem__(self):
#         #pass



if __name__=="__main__":
    dataset = hrc_dataset()
    dataset.create_data()