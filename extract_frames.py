import cv2
import os

def extract_frames(file_directory, output_directory):
  subdir = output_directory.split(os.path.sep)[-1]

  print(output_directory)
  print(file_directory)
  print(subdir)

  # Read the video from specified path 
  cam = cv2.VideoCapture(file_directory) 
  
  # frame 
  currentframe = 0

  while(True): 
        
      # reading from frame 
      ret,frame = cam.read() 
      
      if ret: 
          # if video is still left continue creating images 
          name = str(output_directory) + '/' + str(subdir) + '_' + str(currentframe) + '.jpg'
          print ('Creating... ' + name)
    
          # writing the extracted images 
          cv2.imwrite(name, frame) 
    
          # increasing counter so that it will 
          # show how many frames are created 
          currentframe += 1
      else: 
          print("no ret")
          break
    
  # Release all space and windows once done 
  cam.release() 
  cv2.destroyAllWindows() 