import torchvision
import os
import torch
from einops import rearrange, repeat
import skvideo.io  
import cv2

"""
Utils to support fusion of frames
"""




#for video, this depicrated
def extract_frames_1(ids,frame_list,path="/deepstore/datasets/dmb/MachineLearning/HRC/HRC_files/UCF_Videos"):
    """
        extract_frames(video_path,frame_list=[])
        returns tensor indexed by frame_list variable

        params
        =========
        video_path -> takes in full video path

    """


    getVideoFiles = list(map(lambda x:x[0],ids))
      

    videoDir = list(map(lambda x:x.split("_")[0][:-3] ,getVideoFiles))

    videofile = list(map(lambda x: x.split("_")[0]+"_x264.mp4", getVideoFiles))
  
    

    extracted_frames = None
    for index,value in enumerate(videofile):
        # print(frame_list[index])
        video_path = os.path.join(path,videoDir[index],value)
        #using torchvision
        video = torchvision.io.read_video(video_path)[0]
    
        video = rearrange(video,"f h w c -> f c h w")
        # print(videodata.shape)
        if not isinstance(extracted_frames,torch.Tensor):
            extracted_frames = torch.cat(list(map(lambda x: video[x],[frame_list[index].long()]))).unsqueeze(0)
        else:
            all_frames = torch.cat(list(map(lambda x: video[x],[frame_list[index].long()]))).unsqueeze(0)
            extracted_frames = torch.cat((extracted_frames,all_frames),dim=0)
        print(f"Loaded {value}")

    return extracted_frames/255

#using opencv much faster
def extract_frames(ids,frame_list,trajectories,path="/deepstore/datasets/dmb/MachineLearning/HRC/HRC_files/UCF_Videos"):
    getVideoFiles = list(map(lambda x:x[0],ids))

    videoDir = list(map(lambda x:x.split("_")[0][:-3] ,getVideoFiles))

    videofile = list(map(lambda x: x.split("_")[0]+"_x264.mp4", getVideoFiles))
    
    video_path = os.path.join(path,videoDir[0],videofile[0])
    cap = cv2.VideoCapture(video_path) 

    middle_frame = frame_list[len(frame_list)//2].item()
    
    cap.set(1,middle_frame) 

    ret, frame = cap.read() # Read the frame
    print(frame.shape)

if __name__=="__main__":
    print(extract_frames(ids=[["Assault048_0058"]],frame_list=torch.Tensor([1,2,3,4,5,6,7,10,200,300]).long(),trajectories=[]))