import os
import torch
import cv2 as cv
import numpy as np
import argparse
import re
from extract_frames import extract_frames
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import csv




'''
parser = argparse.ArgumentParser(description='Functions for Visualisation of Skeletons.')

parser.add_argument('frames', type=str, help='Directory containing video frames.')
parser.add_argument('--trajectories', type=str,
                              help='Directory containing the reconstructed/predicted trajectories of people in '
                                   'the video.')
parser.add_argument('--draw_trajectories_skeleton', action='store_true',
                              help='Whether to draw the reconstructed/predicted skeleton or not.')
parser.add_argument('--draw_trajectories_bounding_box', action='store_true',
                              help='Whether to draw the bounding box of the reconstructed/predicted trajectories '
                                   'or not.')
parser.add_argument('--person_id', type=int, help='Draw only a specific person in the video.')
parser.add_argument('--draw_local_skeleton', action='store_true',
                              help='If specified, draw local skeletons on a white background. It must be used '
                                   'in conjunction with --person_id, since it is only possible to visualise '
                                   'one pair (ground-truth, reconstructed/predicted) of local skeletons.')
parser.add_argument('--trajectories_colour', type=str, choices=['black', 'red'],
                              help='Draw the reconstructed/predicted skeletons and bounding boxes in either '
                                   'black or red. If not specified, colours are automatic assigned to skeletons '
                                   'and bounding boxes.')
parser.add_argument('--write_dir', type=str,
                              help='Directory to write rendered frames. If the specified directory does not '
                                   'exist, it will be created.')
'''
    

def create_scatter_plot(model_type, write_dir, test_video,  sum_scores):
    sum_scores_x = sum_scores[0::2]
    sum_scores_y = sum_scores[1::2]
    
    print(sum_scores_x.shape)
    print('SUM attn scores x:', sum_scores_x)
    
    print(sum_scores_y.shape)
    print('SUM attn scores y:', sum_scores_y)
    
    norm_scores_x = torch.nn.functional.normalize(sum_scores_x, p=1, dim=0)
    norm_scores_y = torch.nn.functional.normalize(sum_scores_y, p=1, dim=0)
    print('norm scores:', norm_scores_x)
    print('normalized sum:', sum(norm_scores_x))
    print('norm scores:', norm_scores_y)
    print('normalized sum:', sum(norm_scores_y))
    
    
    if model_type == 'temporal_2':
        x = ['Nose', 'Left eye', 'Right eye', 'Left ear', 'Right ear', 'Left shoulder', 'Right shoulder',
             'Left elbow', 'Right elbow', 'Left wrist', 'Right wrist', 'Left hip', 'Right hip', 'Left knee',
             'Right knee', 'Left ankle', 'Right ankle']
    elif model_type == 'temporal_4':
        x = ['Torso', 'Elbows', 'Wrists', 'Knees', 'Ankles']

    len(x)
    
    fontsize = 25
    
    fig, ax = plt.subplots(1, 1, figsize=(20, 10))
    plt.scatter(x, norm_scores_x, marker='o', s=50, c='r')
    plt.scatter(x, norm_scores_y, marker='o', s=50, c='b')
    ax.tick_params(axis='both', which='major', labelsize=fontsize)
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    
    # Shrink current axis's height by 10% on the bottom
    box = ax.get_position()
    #ax.set_position([box.x0, box.y0 + box.height * 0.1, box.width, box.height * 0.9])
    ax.set_position([box.x0 + box.width * 0.1, box.y0 + box.height * 0.1, box.width * 0.9, box.height * 0.9])
    ax.yaxis.set_label_coords(-.07, 0.5)
    
    plt.ylabel('Attention score', fontsize=fontsize)
    
    labels = ["x coordinate", "y coordinate"]
    
    if model_type == 'temporal_2':
        ax.legend(labels, loc="upper center", bbox_to_anchor=(0.5, -0.3), ncol=2, fancybox=True, fontsize=fontsize)
    elif model_type == 'temporal_4':
        #ax.legend(labels, fontsize=fontsize)
        ax.legend(labels, loc="upper center", bbox_to_anchor=(0.5, -0.15), ncol=2, fancybox=True, fontsize=fontsize)

    #fig.subplots_adjust(bottom=0.5)
    
    image_name =  'scatter_plot_'  + test_video + '_' + model_type + '.jpg'
    image_name = os.path.join(write_dir, image_name)
    plt.savefig(image_name)
    print('scatter plot saved to %s' %(image_name))

def get_colors():
        
    colors = [(255, 0, 0), #blue for torso
            (0, 255, 0), #green for elbow
            (0, 0, 255), #red for wrist
            (255, 0, 127), #purple for knee
            (0, 255, 255) #yellow for ankle
            ] #colors in BGR 
    return colors

def draw_body_parts(frame, keypoints, dotted=False, attn_weight=None):
    
    colors = get_colors() 
    
    #print('keypoints type', type(keypoints))
    #print('\nkeypoints shape', keypoints.shape)
    #print('keypoints', keypoints)
    kp_torso_1 = keypoints[0:7] #joints 0,1,2,3,4,5,6 (head and shoulders) 
    kp_torso_2 = keypoints[11:13] #joints 11,12 (hips)
    
    #print('kp_torso_1', kp_torso_1)
    #print('kp_torso_2', kp_torso_2)
    
    kp_torso = np.concatenate((kp_torso_1, kp_torso_2), axis=0)
    #print('kp_torso type', type(kp_torso))
    #print('kp_torso', kp_torso)
    
    
    kp_elbow = keypoints[7:9]
    kp_wrist = keypoints[9:11]
    kp_knee = keypoints[13:15]
    kp_ankle = keypoints[15:17]
    
    '''
    print('kp_elbow', kp_elbow)
    print('kp_wrist', kp_wrist)
    print('kp_knee', kp_knee)
    print('kp_ankle', kp_ankle)
    '''
    
    kp_torso = np.mean(kp_torso, axis=0)
    kp_elbow = np.mean(kp_elbow, axis=0)
    kp_wrist = np.mean(kp_wrist, axis=0)
    kp_knee = np.mean(kp_knee, axis=0)
    kp_ankle = np.mean(kp_ankle, axis=0)
    
    '''
    print('kp_torso', kp_torso)
    print('kp_elbow', kp_elbow)
    print('kp_wrist', kp_wrist)
    print('kp_knee', kp_knee)
    print('kp_ankle', kp_ankle)
    '''
    
    mean_keypoints = [kp_torso, kp_elbow, kp_wrist, kp_knee, kp_ankle]
    
    #print('mean_keypoints', mean_keypoints)
    
    overlay = frame.copy()
    radius = 5
    
    for index, (x, y) in enumerate(mean_keypoints):
        #print('(x, y)', (x, y))
        if 0 in (x, y):
            continue
        center = int(round(x)), int(round(y))
        cv.circle(overlay, center=center, radius=radius, color=colors[index], thickness=-1)
        cv.circle(frame, center=center, radius=radius, color=colors[index], thickness=1)
   
    if attn_weight is not None:
        alpha = attn_weight # Transparency factor according to the attention weight
    else:
        alpha = 1 #Opaque

    #print('alpha',alpha)
    new_frame = cv.addWeighted(frame, 1-alpha, overlay, alpha, 0)

    return new_frame



def draw_skeleton(frame, keypoints, colour, dotted=False, attn_weight=None, spatial_attn_weight=None, draw_connections=False, draw_grouped_joints=False):
    
    #print('\nDraw skeleton')
    
    connections = [(0, 1), (0, 2), (1, 3), (2, 4),
                   (5, 7), (7, 9), (6, 8), (8, 10),
                   (11, 13), (13, 15), (12, 14), (14, 16),
                   (3, 5), (4, 6), (5, 6), (5, 11), (6, 12), (11, 12)]
    
    overlay = frame.copy()
    
    colors = get_colors()
    #print('keypoints', keypoints)
    
    if spatial_attn_weight is not None:
        
        print('spatial_attn_weights shape', spatial_attn_weight.shape)
        print('spatial_attn_weights', spatial_attn_weight)
        
        spatial_scaler = MinMaxScaler(feature_range=(2, 10))
        scaled_spatial_attn_weight = spatial_attn_weight.reshape(-1, 1)
        scaled_spatial_attn_weight = spatial_scaler.fit_transform(scaled_spatial_attn_weight) #compute min and max, and re-scale mean scores
        scaled_spatial_attn_weight = np.rint(scaled_spatial_attn_weight)
        scaled_spatial_attn_weight = scaled_spatial_attn_weight.astype('int32')
        scaled_spatial_attn_weight = scaled_spatial_attn_weight.reshape(-1)
        
        print('max', spatial_scaler.data_max_)
        print('scaled spatial_attn_weights shape', scaled_spatial_attn_weight.shape)
        print('scaled spatial_attn_weights', scaled_spatial_attn_weight)
    
    
    for index, (x, y) in enumerate(keypoints):
        if 0 in (x, y):
            continue
        center = int(round(x)), int(round(y))
        
        if spatial_attn_weight is not None:
            #print('index', index)
            
            radius = scaled_spatial_attn_weight[index]
        else:
            radius = 4
        
        if draw_grouped_joints:
            #print('index', index)
            if index in [0,1,2,3,4,5,6,11,12]: #torso
                color = colors[0]
            if index in [7,8]: #elbows
                color = colors[1]
            if index in [9,10]: #wrists
                color = colors[2]  
            if index in [13,14]: #knees
                color = colors[3]  
            if index in [15,16]: #ankles
                color = colors[4]  
            #print('color', color)
            cv.circle(overlay, center=center, radius=radius, color=color, thickness=-1)
            #cv.circle(frame, center=center, radius=radius, color=color, thickness=1)
        else:
            cv.circle(overlay, center=center, radius=radius, color=colour, thickness=-1)
            #cv.circle(frame, center=center, radius=radius, color=colour, thickness=1)
    
    if not draw_connections:
        return overlay
    else:
        
        thickness = 2
        
        for keypoint_id1, keypoint_id2 in connections:
            x1, y1 = keypoints[keypoint_id1]
            x2, y2 = keypoints[keypoint_id2]
            if 0 in (x1, y1, x2, y2):
                continue
            pt1 = int(round(x1)), int(round(y1))
            pt2 = int(round(x2)), int(round(y2))
            if dotted:
                #draw_line(frame, pt1=pt1, pt2=pt2, color=colour, thickness=1, gap=5)
                draw_line(overlay, pt1=pt1, pt2=pt2, color=colour, thickness=thickness, gap=5)
            else:
                #imshow(overlay)
                cv.line(overlay, pt1=pt1, pt2=pt2, color=colour, thickness=thickness)
                if attn_weight is not None:
                #if attn_weight>=0:
                  alpha = attn_weight # Transparency factor according to the attention weight
                else:
                  alpha = 1 #Opaque
    
                #print('alpha',alpha)
                new_frame = cv.addWeighted(frame, 1-alpha, overlay, alpha, 0)

        return new_frame


def draw_line(img, pt1, pt2, color, thickness=1, style='dotted', gap=10):
    dist = ((pt1[0] - pt2[0]) ** 2 + (pt1[1] - pt2[1]) ** 2) ** .5
    pts = []
    for i in np.arange(0, dist, gap):
        r = i / dist
        x = int((pt1[0] * (1 - r) + pt2[0] * r) + .5)
        y = int((pt1[1] * (1 - r) + pt2[1] * r) + .5)
        p = (x, y)
        pts.append(p)

    if style == 'dotted':
        for p in pts:
            cv.circle(img, p, thickness, color, -1)
    else:
        s = pts[0]
        e = pts[0]
        i = 0
        for p in pts:
            s = e
            e = p
            if i % 2 == 1:
                cv.line(img, s, e, color, thickness)
            i += 1


def draw_poly(img, pts, color, thickness=1, style='dotted'):
    s = pts[0]
    e = pts[0]
    pts.append(pts.pop(0))
    for p in pts:
        s = e
        e = p
        draw_line(img, s, e, color, thickness, style)


def draw_rect(img, pt1, pt2, color, thickness=1, style='dotted'):
    pts = [pt1, (pt2[0], pt1[1]), pt2, (pt1[0], pt2[1])]
    draw_poly(img, pts, color, thickness, style)


#def render_trajectories_skeletons(args):
#def render_trajectories_skeletons(model_type, test_video, person_id=None, draw_trajectory_segment=False, frames=None, attn_scores=None, spatial_attn_scores=None, spatial_body_part_attn_scores=None):
def render_trajectories_skeletons(model_type, test_video, person_id=None, draw_trajectory_segment=False, frames=None, attn_scores=None, spatial_attn_scores=None):
    '''
    frames_path = args.frames
    trajectories_path = args.trajectories
    draw_trajectories_skeleton = args.draw_trajectories_skeleton
    draw_trajectories_bounding_box = args.draw_trajectories_bounding_box
    specific_person_id = args.person_id
    draw_local_skeleton = args.draw_local_skeleton
    trajectories_colour = args.trajectories_colour
    write_dir = args.write_dir
    '''

    
    #test_video = 'Robbery101'
    test_category = re.split('(\d+)', test_video)[0]

    print('test_category', test_category)
    print('frames', frames)
    #print('attn_scores', attn_scores)

    path = '/data/s3447707/MasterThesis/'

    frames_path = os.path.join(path, 'test_frames', test_video)
    trajectories_path = os.path.join('/data/s3447707/HR-Crime/Trajectories', test_category, test_video)
    #trajectories_path = '/content/gdrive/MyDrive/CAIP HR-Crime/HR-Crime/Trajectories/Robbery/Robbery101'
    draw_trajectories_skeleton = True
    draw_trajectories_bounding_box = False
    specific_person_id = person_id
    specific_frames = frames
    draw_local_skeleton = False
    trajectories_colour = 'red'


    if trajectories_path is None:
        raise ValueError('--trajectories must be specified.')

    if not any([draw_trajectories_skeleton, draw_trajectories_bounding_box]):
        raise ValueError('--draw_trajectories_skeleton or '
                         '--draw_trajectories_bounding_box must be specified.')

    if draw_local_skeleton and specific_person_id is None:
        raise ValueError('If --draw_local_skeleton is specified, a --person_id must be chosen as well.')
    elif draw_local_skeleton:
        draw_trajectories_skeleton = True
        draw_trajectories_bounding_box = False
    
    if (draw_trajectory_segment and specific_frames is None):
      raise ValueError('Must specify --frames to draw when --draw_trajectory_segment is chosen')

    print('specific_frames', specific_frames)
    
    '''
    if (attn_scores is not None and specific_frames is None):
      raise ValueError('Must specify --frames to display --attn_weights')
    '''


    if specific_person_id and specific_frames.any():
      specific_start_frame = specific_frames[0]
      specific_end_frame = specific_frames[-1]
      write_dir = os.path.join(path, 'skeleton_visualizations_specific_person_specific_frames', model_type, test_video, test_video + '_' + 
                               str(specific_person_id) + '_start_frame_' + str(specific_start_frame) + '_end_frame_' + str(specific_end_frame))
    elif specific_person_id:
      write_dir = os.path.join(path, 'skeleton_visualizations_specific_person_specific_frames', model_type, test_video, test_video + '_' + str(specific_person_id))


    print('write_dir',write_dir)
    maybe_create_dir(write_dir)

    _render_trajectories_skeletons(model_type,
                                   write_dir,
                                   test_video,
                                   frames_path, trajectories_path,
                                   draw_trajectories_skeleton,
                                   draw_trajectories_bounding_box,
                                   specific_person_id,
                                   specific_frames,
                                   attn_scores,
                                   spatial_attn_scores,
                                   #spatial_body_part_attn_scores,
                                   draw_local_skeleton,
                                   trajectories_colour)
    print('Visualisation successfully rendered to %s' % write_dir)
    
    if model_type in ['temporal_2', 'temporal_4']:
        print('\nCreate scatter plot\n')
        create_scatter_plot(model_type, write_dir, test_video, attn_scores)

    return None


def _render_trajectories_skeletons(model_type,
                                   write_dir,
                                   test_video,
                                   frames_path, trajectories_path,
                                   draw_trajectories_skeleton,
                                   draw_trajectories_bounding_box,
                                   specific_person_id=None,
                                   specific_frames=None,
                                   attn_scores=None,
                                   spatial_attn_scores=None,
                                   #spatial_body_part_attn_scores=None,
                                   draw_local_skeleton=False,
                                   trajectories_colour=None):
    frames_names = os.listdir(frames_path)
    frames_names.sort(key=lambda f: int(re.sub('\D', '', f)))
    #print('frames_names', frames_names)
    
    if specific_frames.any():
      min_frame_id = specific_frames[0]
      max_frame_id = specific_frames[-1] + 1
    else:
      min_frame_id = 0
      max_frame_id = len(frames_names)

    print('min_frame_id', min_frame_id)
    print('max_frame_id', max_frame_id)
    
    attention_scores_file = os.path.join(write_dir,'attention_scores.csv')
    f = open(attention_scores_file, "w")
    csv_writer = csv.writer(f, delimiter=';')

    if model_type in ('spatial-temporal','parts'):
        csv_writer.writerow(['Frame index', 'Frame', 'Frame attention', 'Spatial attention'])
    else:
        csv_writer.writerow(['Frame index', 'Frame', 'Frame attention'])

    norm_scores = torch.nn.functional.normalize(attn_scores, p=1, dim=0) #normalize scores to sum 1
    print('\nnorm_scores.shape', norm_scores.shape)

    print('norm scores:', norm_scores)
    print('sum norm scores', sum(norm_scores))
    attn_weights = norm_scores.reshape(-1, 1)
    #print('attn_weights', attn_weights)
    
    scaler = MinMaxScaler(feature_range=(0.3, 1))
    attn_weights = scaler.fit_transform(attn_weights) #compute min and max, and re-scale mean scores
    attn_weights = attn_weights.reshape(-1)
    #print('scaled attn_weights', attn_weights)
    
    
    if model_type in ('spatial-temporal','parts'):
        norm_spatial_scores = torch.nn.functional.normalize(spatial_attn_scores, p=1, dim=1) #normalize scores to sum 1
        print('\n\nnorm_spatial_scores shape', norm_spatial_scores.shape)
        #print('norm_spatial_scores:', norm_spatial_scores)
        #print(sum(norm_spatial_scores))
        print('sum norm_spatial_scores', torch.sum(norm_spatial_scores, dim=1))
        '''
        spatial_attn_weights = norm_spatial_scores
        print('spatial_attn_weights shape', spatial_attn_weights.shape)
        
        spatial_scaler = MinMaxScaler(feature_range=(2, 10))
        spatial_attn_weights = spatial_scaler.fit_transform(spatial_attn_weights) #compute min and max, and re-scale mean scores
        spatial_attn_weights = np.rint(spatial_attn_weights)
        spatial_attn_weights = spatial_attn_weights.astype('int32')
        print('max', spatial_scaler.data_max_)
        print('scaled spatial_attn_weights shape', spatial_attn_weights.shape)
        '''
    
    rendered_frames = {}
    if trajectories_path is not None:
        trajectories_files_names = sorted(os.listdir(trajectories_path))  # 001.csv, 002.csv, ...
        #print(trajectories_files_names)

        
        for trajectory_file_name in trajectories_files_names:
            person_id = int(trajectory_file_name.split('.')[0])
            #print('person_id',person_id)
            #print('specific_person_id',specific_person_id)
            if specific_person_id is not None and specific_person_id != person_id:
                continue
            
            '''
            if trajectories_colour is None:
                colour = COLOURS[person_id % len(COLOURS)]
            else:
                colour = (0, 0, 0) if trajectories_colour == 'black' else (0, 0, 255) #color in BGR
            '''
            colour = (0, 0, 0) if trajectories_colour == 'black' else (0, 0, 255) #color in BGR

            trajectory = np.loadtxt(os.path.join(trajectories_path, trajectory_file_name), delimiter=',', ndmin=2)
            trajectory_frames = trajectory[:, 0].astype(np.int32)
            trajectory_coordinates = trajectory[:, 1:]

            for frame_id, skeleton_coordinates in zip(trajectory_frames, trajectory_coordinates):
                if frame_id >= max_frame_id:
                    break

                if frame_id < min_frame_id:
                  continue
                
                
                frame_index = np.where(specific_frames == frame_id)[0][0]
                print('frame_id', frame_id)
                print('frame_index', frame_index)
                frame = rendered_frames.get(frame_id)
                if frame is None:
                    frame = cv.imread(os.path.join(frames_path, frames_names[frame_id]))
                    if draw_local_skeleton:
                        frame = np.full_like(frame, fill_value=255)

                if draw_trajectories_skeleton:
                    if draw_local_skeleton:
                        height, width = frame.shape[:2]
                        left, right, top, bottom = compute_simple_bounding_box(skeleton_coordinates)
                        bb_center = np.array([(left + right) / 2, (top + bottom) / 2], dtype=np.float32)
                        target_center = np.array([3 * width / 4, height / 2], dtype=np.float32)
                        displacement_vector = target_center - bb_center
                        frame = draw_skeleton(frame, keypoints=skeleton_coordinates.reshape(-1, 2) + displacement_vector,
                                      colour=colour, dotted=False, attn_weight= attn_weights[frame_index])
                    else:
                        if model_type == 'temporal':
                            print('frame attn weight', attn_weights[frame_index])
                            frame = draw_skeleton(frame, keypoints=skeleton_coordinates.reshape(-1, 2), colour=colour, dotted=False, attn_weight=attn_weights[frame_index], draw_connections=True)
                        elif model_type == 'temporal_2':
                            frame = draw_skeleton(frame, keypoints=skeleton_coordinates.reshape(-1, 2), colour=colour, dotted=False)
                        elif model_type == 'temporal_3':
                            print('frame attn weight', attn_weights[frame_index])
                            frame = draw_body_parts(frame, keypoints=skeleton_coordinates.reshape(-1, 2), dotted=False, attn_weight=attn_weights[frame_index])
                        elif model_type == 'temporal_4':
                            frame = draw_body_parts(frame, keypoints=skeleton_coordinates.reshape(-1, 2), dotted=False)
                        elif model_type == 'spatial-temporal':
                            print('frame norm_scores', norm_scores[frame_index])
                            print('frame attn weight', attn_weights[frame_index])
                            print('frame norm_spatial_scores', norm_spatial_scores[frame_index, :])
                            #print('frame spatial_attn_weights', spatial_attn_weights[frame_index, :])
                            #frame = draw_skeleton(frame, keypoints=skeleton_coordinates.reshape(-1, 2),
                            #                      colour=colour, attn_weight=attn_weights[frame_index],
                            #                      spatial_attn_weight=spatial_attn_weights[frame_index,:], draw_connections=True)
                            frame = draw_skeleton(frame, keypoints=skeleton_coordinates.reshape(-1, 2),
                                                  colour=colour, attn_weight=attn_weights[frame_index],
                                                  spatial_attn_weight=norm_spatial_scores[frame_index,:], draw_connections=True)
                        elif model_type == 'parts':
                            print('frame norm_scores', norm_scores[frame_index])
                            print('frame attn weight', attn_weights[frame_index])
                            print('frame norm_spatial_body_part_scores', norm_spatial_scores[frame_index, :])
                            #print('frame spatial_body_part_attn_weights', spatial_attn_weights[frame_index, :])
                            #frame = draw_skeleton(frame, keypoints=skeleton_coordinates.reshape(-1, 2), 
                            #                      colour=(0, 0, 0), attn_weight=attn_weights[frame_index],
                            #                      spatial_attn_weight=spatial_attn_weights[frame_index,:], draw_connections=True, draw_grouped_joints=True)
                            frame = draw_skeleton(frame, keypoints=skeleton_coordinates.reshape(-1, 2), 
                                                  colour=(0, 0, 0), attn_weight=attn_weights[frame_index],
                                                  spatial_attn_weight=norm_spatial_scores[frame_index,:], draw_connections=True, draw_grouped_joints=True)
                            
                            

                if draw_trajectories_bounding_box:
                    left, right, top, bottom = compute_simple_bounding_box(skeleton_coordinates)
                    bb_center = int(round((left + right) / 2)), int(round((top + bottom) / 2))
                    cv.circle(frame, center=bb_center, radius=4, color=colour, thickness=-1)
                    draw_rect(frame, pt1=(left, top), pt2=(right, bottom), color=colour, thickness=3, style='dotted')

                #print('frame_id', frame_id)
                rendered_frames[frame_id] = frame
    

    for frame_id, frame_name in enumerate(frames_names):     
        if frame_id >= max_frame_id:
          break
        
        if (frame_id < min_frame_id) or (frame_id not in specific_frames):
           continue  
        
        print('frame_id', frame_id)
        frame = rendered_frames.get(frame_id)
        if frame is None:
            frame = cv.imread(os.path.join(frames_path, frame_name))
            if draw_local_skeleton:
                frame = np.full_like(frame, fill_value=255)
        
        
        frame_name, _ = frame_name.split('.')
        frame_index = np.where(specific_frames == frame_id)[0][0]
        print('frame_index', frame_index)
        if model_type in ['temporal', 'temporal_3', 'spatial-temporal','parts']: #only save scores in the file name for sequences of frames
            frame_name = frame_name + '_score_' + str(round(norm_scores[frame_index].item(),4)) + '.jpg'
        elif model_type in ['temporal_2', 'temporal_4']: #do not save scores in the file name for sequences of joints/body parts
            frame_name = frame_name + '.jpg'
         
        print('frame_name', frame_name)

        #np.set_printoptions(suppress=True)
        if model_type in ('spatial-temporal','parts'):
            scores_list = norm_spatial_scores[frame_index, :].numpy()
            csv_writer.writerow([frame_index, frame_name, 
                                 norm_scores[frame_index].numpy(), 
                                 sorted(range(len(scores_list)), key=lambda i: scores_list[i], reverse=True)[:5],
                                 sorted(scores_list, reverse=True)[:5],
                                 scores_list])
        else:
            csv_writer.writerow([frame_index, frame_name, norm_scores[frame_index].numpy()])
        
        image_dir = os.path.join(write_dir, frame_name) #save image (.jpg)
        print('image_dir: ',image_dir)
        cv.imwrite(image_dir, img=frame)

    f.close()
    print('\nAttention scores written to', attention_scores_file)


def maybe_create_dir(dir_path):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
        return True
    
    print('Directory %s already exists' % (dir_path))

    return False


def compute_simple_bounding_box(skeleton):
    x = skeleton[::2]
    x = np.where(x == 0.0, np.nan, x)
    left, right = int(round(np.nanmin(x))), int(round(np.nanmax(x)))
    y = skeleton[1::2]
    y = np.where(y == 0.0, np.nan, y)
    top, bottom = int(round(np.nanmin(y))), int(round(np.nanmax(y)))

    return left, right, top, bottom


#def visualize_skeleton_and_attention(model_type, test_video, person_id, draw_trajectory_segment, test_frames, scores, spatial_scores=None, spatial_body_part_scores=None):
def visualize_skeleton_and_attention(model_type, test_video, person_id, draw_trajectory_segment, test_frames, scores, spatial_scores=None):    
    
    video_path = '/data/s3447707/MasterThesis/test_videos/'
    frames_path = '/data/s3447707/MasterThesis/test_frames/'
    
    video_file = os.path.join(video_path, test_video) + '_x264.mp4'
    output_directory = os.path.join(frames_path, test_video)
    
    
    if not os.path.exists(output_directory):
        # create a subfolder for the video
        print('\nmake new directory ' + str(output_directory))
        os.makedirs(output_directory)
    
    #print('list output_directory', os.listdir(output_directory))
    
    if os.listdir(output_directory)==[]:
        print('\nextract frames')
        extract_frames(video_file,output_directory)
    
    person_id = int(person_id)
    draw_trajectory_segment = True
    
    scores = scores[1:]
    
    print('\n\nscores shape', scores.shape)
    print('\n\nscores', scores)
    
    
    #if model_type == 'spatial-temporal':
    if model_type in ('spatial-temporal','parts'):
        print('render trajcetories using spatial_scores with shape', spatial_scores.shape)
        #print('spatial_scores', spatial_scores)
        render_trajectories_skeletons(model_type, test_video, person_id, draw_trajectory_segment, test_frames, attn_scores=scores, spatial_attn_scores=spatial_scores)
    else:
        render_trajectories_skeletons(model_type, test_video, person_id, draw_trajectory_segment, test_frames, attn_scores=scores)
    
    '''
    elif model_type == 'parts':
        print('spatial_body_part_scores shape', spatial_body_part_scores.shape)
        #print('spatial_body_part_scores', spatial_body_part_scores)
        render_trajectories_skeletons(model_type, test_video, person_id, draw_trajectory_segment, test_frames, attn_scores=scores, spatial_body_part_attn_scores=spatial_body_part_scores)
    '''