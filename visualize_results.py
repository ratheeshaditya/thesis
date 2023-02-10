'''
VISUALIZING SCRIPT
'''


'''
Import packages
'''

import torch
import torch.nn.functional as F

import numpy as np

from trajectory import Trajectory, extract_fixed_sized_segments, split_into_train_and_test, remove_short_trajectories, get_categories, get_UTK_categories
from transformer_store_attn import TubeletTemporalSpatialPart_concat_chan_2_Transformer_store_attn, TubeletTemporalPart_concat_chan_1_Transformer_store_attn, TubeletTemporalTransformer_store_attn, TubeletTemporalPart_mean_chan_1_Transformer_store_attn, TubeletTemporalPart_mean_chan_2_Transformer_store_attn, TubeletTemporalPart_concat_chan_2_Transformer_store_attn, TemporalTransformer_4_store_attn, TemporalTransformer_3_store_attn, TemporalTransformer_2_store_attn, BodyPartTransformer_store_attn, SpatialTemporalTransformer_store_attn, TemporalTransformer_store_attn
from trajectory import TrajectoryDataset, Trajectory, extract_fixed_sized_segments, split_into_train_and_test, remove_short_trajectories, get_NTU_categories, get_categories
import pickle
import os
import argparse
import yaml
from einops import rearrange
from utils import print_statistics, SetupLogger, evaluate_all, evaluate_category, conv_to_float, SetupFolders, train_acc, SetupVisFolders

from visualize_attention_weights import visualize_attention_weights
from visualize_skeleton_and_attention import visualize_skeleton_and_attention
#from extract_frames import extract_frames


parser = argparse.ArgumentParser()
parser.add_argument("--config_file", help="file from which configs need to be loaded", default="config")
args = parser.parse_args()

with open(args.config_file, "r") as ymlfile:
    cfg = yaml.load(ymlfile, Loader=yaml.FullLoader)

base_folder, log_dir = SetupVisFolders(cfg['META']['NAME'], cfg['MODEL']['DATASET'], cfg['VIS']['TYPE'])

logger = SetupLogger('logger', log_dir)
logger.info("Logger set up!")
logger.info("Tensorboard set up!\n\n\n\n")
logger.info("FOLDER NAME: %s", cfg['META']['NAME'])
logger.info("\nCONFIGS \n=======\n"+yaml.dump(cfg))

with open(os.path.join(base_folder,'config.yml'), 'w') as config_file:
    yaml.dump(cfg, config_file)

# set parameters
model_type = cfg['MODEL']['MODEL_TYPE']
filename = cfg['META']['NAME']
embed_dim = cfg['MODEL']['EMBED_DIM']
test_sample = cfg['VIS']['TEST_SAMPLE']
test_index = cfg['VIS']['TEST_INDEX']
segment_length = cfg['MODEL']['SEGMENT_LEN']
dataset = cfg['MODEL']['DATASET']
vis_type = cfg['VIS']['TYPE']


if dataset == "HRC":
    num_classes = 13
    num_joints = 17
    num_parts = 5
    in_chans = 2
elif dataset == "UTK":
    num_classes = 10
    num_joints = 20
    in_chans = 3
elif "NTU" in dataset:
    if "2D" in dataset:
        num_classes = 120
        num_joints = 25
        num_parts = 5
        in_chans = 2
    elif "3D" in dataset:
        num_classes = 120
        num_joints = 25
        num_parts = 5
        in_chans = 3

if cfg['DECOMPOSED']['ENABLE']:
        if cfg['DECOMPOSED']['TYPE'] == "GR":
            num_joints = num_joints*2
        elif cfg['DECOMPOSED']['TYPE'] == "GS":
            num_joints+=1

#test parameters
#filename = 'FINAL_train_temporal_transformer_model_cross_val_on_gpu_embed_dim_256_reserve_1_day_fold_3'
#model_type = 'temporal'
#embed_dim = 256
#test_sample = 'Abuse012_0002'

PATH = os.path.join('/home/s2435462/HRC/results/', dataset, filename, 'models', filename+'_fold_1.pt')
# PATH = '/home/s2435462/HRC/results/'+dataset+'/NTU_2D_ttpcc1/models'
# PATH = '/data/s3447707/MasterThesis/trained_models/' + filename + '.pt'
model_ = torch.load(PATH)


NEW_PATH = base_folder +'/'+ filename + 'state_dict.pt'
if not os.path.isfile(NEW_PATH):
    torch.save(model_.state_dict(), NEW_PATH)
    print('\nsave model state dict to', NEW_PATH)
else:
    print('\model state dict %s already exists' % (NEW_PATH))


#Create model object
if cfg['MODEL']['MODEL_TYPE'] == 'temporal':
    model = TemporalTransformer_store_attn(embed_dim=embed_dim, num_frames=segment_length, num_classes=num_classes, num_joints=num_joints, in_chans=in_chans, mlp_ratio=2., qkv_bias=True, qk_scale=None, dropout=0.1)
elif cfg['MODEL']['MODEL_TYPE'] == 'temporal_2':
    model = TemporalTransformer_2_store_attn(embed_dim=embed_dim, num_frames=segment_length, num_classes=num_classes, num_joints=num_joints, in_chans=in_chans, mlp_ratio=2., qkv_bias=True, qk_scale=None, dropout=0.1)
elif cfg['MODEL']['MODEL_TYPE'] == 'temporal_3':
    model = TemporalTransformer_3_store_attn(embed_dim=embed_dim, num_frames=segment_length, num_classes=num_classes, num_joints=num_joints, num_parts=num_parts, in_chans=in_chans, mlp_ratio=2., qkv_bias=True, qk_scale=None, dropout=0.1)
elif cfg['MODEL']['MODEL_TYPE'] == 'temporal_4':
    model = TemporalTransformer_4_store_attn(embed_dim=embed_dim, num_frames=segment_length, num_classes=num_classes, num_joints=num_joints, num_parts=num_parts, in_chans=in_chans, mlp_ratio=2., qkv_bias=True, qk_scale=None, dropout=0.1)
elif cfg['MODEL']['MODEL_TYPE'] == 'spatial-temporal':
    model = SpatialTemporalTransformer_store_attn(embed_dim_ratio=embed_dim, num_frames=segment_length, num_classes=num_classes, num_joints=num_joints, in_chans=in_chans, mlp_ratio=2., qkv_bias=True, qk_scale=None, dropout=0.1)
elif cfg['MODEL']['MODEL_TYPE'] == "parts":
    model = BodyPartTransformer_store_attn(dataset=dataset, embed_dim_ratio=embed_dim, num_frames=segment_length, num_classes=num_classes, num_joints=num_joints, in_chans=in_chans, mlp_ratio=2., qkv_bias=True, qk_scale=None, dropout=0.1)
elif cfg['MODEL']['MODEL_TYPE'] == "tubelet_temporal":
    kernel = tuple(map(int, cfg['TUBELET']['KERNEL'].split(',')))
    stride = tuple(map(int, cfg['TUBELET']['STRIDE'].split(',')))
    model = TubeletTemporalTransformer_store_attn(dataset=dataset, embed_dim=embed_dim, num_frames=segment_length, num_classes=num_classes, num_joints=num_joints, in_chans=in_chans, kernel=kernel, stride=stride, mlp_ratio=2., qkv_bias=True, qk_scale=None, dropout=0.1)
elif cfg['MODEL']['MODEL_TYPE'] == "ttpmc1":
    kernel = tuple(map(int, cfg['TUBELET']['KERNEL'].split(',')))
    stride = tuple(map(int, cfg['TUBELET']['STRIDE'].split(',')))
    model = TubeletTemporalPart_mean_chan_1_Transformer_store_attn(dataset=dataset, embed_dim=embed_dim, num_frames=segment_length, num_classes=num_classes, num_joints=num_joints, in_chans=in_chans, kernel=kernel, stride=stride, mlp_ratio=2., qkv_bias=True, qk_scale=None, dropout=0.1)
elif cfg['MODEL']['MODEL_TYPE'] == "ttpcc1":
    kernel = tuple(map(int, cfg['TUBELET']['KERNEL'].split(',')))
    stride = tuple(map(int, cfg['TUBELET']['STRIDE'].split(',')))
    model = TubeletTemporalPart_concat_chan_1_Transformer_store_attn(dataset=dataset, embed_dim=embed_dim, num_frames=segment_length, num_classes=num_classes, num_joints=num_joints, in_chans=in_chans, kernel=kernel, stride=stride, mlp_ratio=2., qkv_bias=True, qk_scale=None, dropout=0.1)
elif cfg['MODEL']['MODEL_TYPE'] == "ttpmc2":
    kernel = tuple(map(int, cfg['TUBELET']['KERNEL'].split(',')))
    stride = tuple(map(int, cfg['TUBELET']['STRIDE'].split(',')))
    model = TubeletTemporalPart_mean_chan_2_Transformer_store_attn(dataset=dataset, embed_dim=embed_dim, num_frames=segment_length, num_classes=num_classes, num_joints=num_joints, in_chans=in_chans, kernel=kernel, stride=stride, mlp_ratio=2., qkv_bias=True, qk_scale=None, dropout=0.1)
elif cfg['MODEL']['MODEL_TYPE'] == "ttpcc2":
    kernel = tuple(map(int, cfg['TUBELET']['KERNEL'].split(',')))
    stride = tuple(map(int, cfg['TUBELET']['STRIDE'].split(',')))
    model = TubeletTemporalPart_concat_chan_2_Transformer_store_attn(dataset=dataset, embed_dim=embed_dim, num_frames=segment_length, num_classes=num_classes, num_joints=num_joints, in_chans=in_chans, kernel=kernel, stride=stride, mlp_ratio=2., qkv_bias=True, qk_scale=None, dropout=0.1)
elif cfg['MODEL']['MODEL_TYPE'] == "ttspcc2":
    kernel = tuple(map(int, cfg['TUBELET']['KERNEL'].split(',')))
    stride = tuple(map(int, cfg['TUBELET']['STRIDE'].split(',')))
    model = TubeletTemporalSpatialPart_concat_chan_2_Transformer_store_attn(dataset=dataset, embed_dim_ratio=embed_dim, num_frames=segment_length, num_classes=num_classes, num_joints=num_joints, in_chans=in_chans, kernel=kernel, stride=stride, mlp_ratio=2., qkv_bias=True, qk_scale=None, dropout=0.1, pad_mode = cfg['TUBELET']['PAD_MODE'])


#Load model state dict
model.load_state_dict(torch.load(NEW_PATH), strict=False)
model.eval()

if cfg['DECOMPOSED']['ENABLE']:
    if cfg['DECOMPOSED']['TYPE'] == "GR":
        dec_GR_path = "decom_GR_"
    elif cfg['DECOMPOSED']['TYPE'] == "GS":
        dec_GR_path = "decom_"

if dataset=="HRC":
    decomposed = dec_GR_path if cfg['DECOMPOSED']['ENABLE'] else ""
    dimension = "2D"

    PIK_train = "/home/s2435462/HRC/data/"+dataset+"/trajectories_train_HRC_"+decomposed+dimension+".dat"
    PIK_test = "/home/s2435462/HRC/data/"+dataset+"/trajectories_test_HRC_"+decomposed+dimension+".dat"

    all_categories = get_categories()
elif dataset == "UTK":
    PIK_train = "./data/train_UTK_trajectories.dat"
    PIK_test = "./data/test_UTK_trajectories.dat"
    all_categories = get_UTK_categories()
elif "NTU" in dataset:
    dimension = dataset.split('_')[-1]
    decomposed = dec_GR_path if cfg['DECOMPOSED']['ENABLE'] else ""

    PIK_train = "/home/s2435462/HRC/data/"+dataset+"/trajectories_train_NTU_"+decomposed+dimension+".dat"
    PIK_test = "/home/s2435462/HRC/data/"+dataset+"/trajectories_test_NTU_"+decomposed+dimension+".dat"
    all_categories = get_NTU_categories()
else:
    raise Exception('dataset not recognized, must be HRC or NTU')


if dataset == 'HRC':
    with open(PIK_train, "rb") as f:
        test_crime_trajectories = pickle.load(f)
else:
    if cfg['DECOMPOSED']['ENABLE']:
        with open(PIK_train, "rb") as f:
            test_crime_trajectories = pickle.load(f)
    else:
        with open(PIK_test, "rb") as f:
            test_crime_trajectories = pickle.load(f)



print('\nLoaded %d test trajectories' % (len(test_crime_trajectories)))

def collator_for_lists(batch):
        '''
        Reference : https://stackoverflow.com/questions/64883998/pytorch-dataloader-shows-odd-behavior-with-string-dataset
        Reference : https://stackoverflow.com/questions/52818145/why-pytorch-dataloader-behaves-differently-on-numpy-array-and-list
        '''
        # assert all('sentences' in x for x in batch)
        # assert all('label' in x for x in batch)
        return {
            'id': [x['id'] for x in batch],
            'videos': [x['videos'] for x in batch],
            'persons': [x['persons'] for x in batch],
            'frames': torch.tensor(np.array([x['frames'] for x in batch])),
            'categories': torch.tensor(np.array([x['categories'] for x in batch])),
            'coordinates': torch.tensor(np.array([x['coordinates'] for x in batch]))
        }


test_sample_trajectory= {}

test_sample_trajectory[test_sample] = test_crime_trajectories[test_sample]

print(test_sample_trajectory)

print('\nTest sample %s has %d frames' % (test_sample, len(test_sample_trajectory[test_sample].frames)))

test = TrajectoryDataset(*extract_fixed_sized_segments(dataset, test_sample_trajectory, input_length=segment_length))

number_of_segments = len(test)
print('number_of_segments', number_of_segments)

test_dataloader = torch.utils.data.DataLoader(test, batch_size = number_of_segments, shuffle=False, collate_fn=collator_for_lists)

batch = next(iter(test_dataloader))

ids, videos, persons, frames, data, categories = batch['id'], batch['videos'], batch['persons'], batch['frames'], batch['coordinates'], batch['categories']

device = 'cpu'

labels = torch.tensor([y[0] for y in categories]).to(device)
# videos = videos.to(device)
# persons = persons.to(device)
frames = frames.to(device)
data = data.to(device)

print('labels length:', len(labels))
print('videos:', videos)
print(frames)
print('frames at index %d: %s' % (test_index, frames[test_index]))

outputs = model(data)


_, predictions = torch.max(outputs, dim=1)
print('true label', labels[test_index])
print('prediction', predictions[test_index])

#Visualize attention weights
if vis_type == 'ATTN':
    
    image_location = '/home/s2435462/HRC/results/images_attn_weights'
    image_location = os.path.join(image_location, dataset)
    image_location = os.path.join(image_location, model_type)

    image_location = base_folder
    
    if not os.path.exists(image_location):
        # create a subfolder for the model
        print('Making new directory ' + str(image_location))
        os.makedirs(image_location)
    
    visualize_attention_weights(model_type, model, test_sample, test_index, image_location, filename)


#Visualize skeleton and attention_weights
if vis_type == 'SKEL':
    test_video, person_id = test_sample.split('_')
    
    '''
    video_path = '/data/s3447707/MasterThesis/test_videos/'
    frames_path = '/data/s3447707/MasterThesis/test_frames/'
    
    video_file = os.path.join(video_path, test_video, '_x264.mp4')
    output_directory = os.path.join(frames_path, test_video)
    
    
    if not os.path.exists(output_directory):
        # create a subfolder for the video
        print('make new directory ' + str(output_directory))
        os.makedirs(output_directory)

        if os.listdir(output_directory)==[]:
            extract_frames(video_file,output_directory)
    '''
    
    '''
    if not os.path.exists(file_name_test):
        print('Generate new test results with frames')
        file, fold_number = filename.rsplit('_fold_',1)
        print('file', file)
        print('fold_number', fold_number)
        #python save_outputs_with_frames.py --filename FINAL_temporal_transformer_embed_dim_256_segment_length_24 --fold_number 2 --embed_dim 256 --model_type temporal --segment_length 24 --debug
    else:
        print('Frames already extracted for test video %s' % test_video)
    '''
    
    
    class_token_index = 0
    num_layers = len(model.blocks)
    layer = num_layers-1
    print('layer', layer)
    
    #image_location = '/data/s3447707/MasterThesis/images_attention_weights/'
    
    #average attention weights of last layer for the class token to all other elements in the sequence (first row in attention matrix).
    scores = model.blocks[layer].attn.attn_scores[test_index,:,class_token_index,:].data
    print('scores.shape', scores.shape)
    #print('attn scores:', scores)
    
    sum_scores = torch.sum(scores, 0)
    print('sum_scores.shape', sum_scores.shape)
    print('SUM attn scores:', sum_scores)
    
    draw_trajectory_segment = True
    # test_frames = traj_frames_test[test_index]
    test_frames = frames[test_index]
    
    if model_type == 'spatial-temporal':
        
        print('\n\n\nRetrieve spatial block attention scores')
        spatial_scores = model.Spatial_blocks[layer].attn.attn_scores.data
        print('spatial_scores shape', spatial_scores.shape)
        
        #must first rearange attention scores back to [batch_size, num_frames, num_heads, num_joint, num_joint], dimensions 0, 1, 2, 3, 4
        spatial_scores = rearrange(spatial_scores, '(b f) h j1 j2  -> b f h j1 j2', f=segment_length) 
        print('rearranged shape', spatial_scores.shape)
        
        #we do not use the class token in the spatial transformer module, we therefore have to sum the attention scores over all rows (dimension 3) and then over all heads (dimension 2)
        spatial_scores = spatial_scores[test_index,:,:,:,:].data #has shape [num_frames, num_heads, num_joints, num_joints], e.g. [12, 8, 17, 17]
        print('spatial_scores.shape', spatial_scores.shape)
        print('spatial_scores[0,0,:,0]:', spatial_scores[0,0,:,0])
        
        sum_spatial_scores = torch.sum(spatial_scores, 2) #sum over rows of joints
        print('sum_spatial_scores.shape', sum_spatial_scores.shape)
        print('sum_spatial_scores[0,:,0]:', sum_spatial_scores[0,:,0])
        
        sum_spatial_scores = torch.sum(sum_spatial_scores, 1) #sum over rows of joints
        print('SUM sum_spatial_scores.shape', sum_spatial_scores.shape)
        print('SUM sum_spatial_scores[0]:', sum_spatial_scores[0])
        
        visualize_skeleton_and_attention(model_type, test_video, person_id, draw_trajectory_segment, test_frames, scores=sum_scores, spatial_scores=sum_spatial_scores)
        
    elif model_type == 'parts':
        
        print('\n\n\nRetrieve body part attention scores')
        torso_scores = model.Torso_blocks[layer].attn.attn_scores.data
        elbow_scores = model.Elbow_blocks[layer].attn.attn_scores.data
        wrist_scores = model.Wrist_blocks[layer].attn.attn_scores.data
        knee_scores = model.Knee_blocks[layer].attn.attn_scores.data
        ankle_scores = model.Ankle_blocks[layer].attn.attn_scores.data
        print('torso_scores shape', torso_scores.shape)
        print('elbow_scores shape', elbow_scores.shape)
        print('wrist_scores shape', wrist_scores.shape)
        print('knee_scores shape', knee_scores.shape)
        print('ankle_scores shape', ankle_scores.shape)
        
        
        #must first rearange attention scores back to [batch_size, num_frames, num_heads, num_joint, num_joint], dimensions 0, 1, 2, 3, 4
        torso_scores = rearrange(torso_scores, '(b f) h j1 j2  -> b f h j1 j2', f=segment_length) 
        elbow_scores = rearrange(elbow_scores, '(b f) h j1 j2  -> b f h j1 j2', f=segment_length) 
        wrist_scores = rearrange(wrist_scores, '(b f) h j1 j2  -> b f h j1 j2', f=segment_length) 
        knee_scores = rearrange(knee_scores, '(b f) h j1 j2  -> b f h j1 j2', f=segment_length) 
        ankle_scores = rearrange(ankle_scores, '(b f) h j1 j2  -> b f h j1 j2', f=segment_length) 
        print('rearranged shape', torso_scores.shape)
        print('rearranged shape', elbow_scores.shape)
        print('rearranged shape', wrist_scores.shape)
        print('rearranged shape', knee_scores.shape)
        print('rearranged shape', ankle_scores.shape)
        
        #we do not use the class token in the body part transformer module, we therefore have to sum the attention scores over all rows (dimension 3) and then over all heads (dimension 2)
        torso_scores = torso_scores[test_index,:,:,:,:].data #has shape [num_frames, num_heads, num_joints, num_joints], e.g. [12, 8, 17, 17]
        elbow_scores = elbow_scores[test_index,:,:,:,:].data #has shape [num_frames, num_heads, num_joints, num_joints], e.g. [12, 8, 17, 17]
        wrist_scores = wrist_scores[test_index,:,:,:,:].data #has shape [num_frames, num_heads, num_joints, num_joints], e.g. [12, 8, 17, 17]
        knee_scores = knee_scores[test_index,:,:,:,:].data #has shape [num_frames, num_heads, num_joints, num_joints], e.g. [12, 8, 17, 17]
        ankle_scores = ankle_scores[test_index,:,:,:,:].data #has shape [num_frames, num_heads, num_joints, num_joints], e.g. [12, 8, 17, 17]
        print('torso_scores.shape', torso_scores.shape)
        print('elbow_scores.shape', elbow_scores.shape)
        print('wrist_scores.shape', wrist_scores.shape)
        print('knee_scores.shape', knee_scores.shape)
        print('ankle_scores.shape', ankle_scores.shape)
        #print('torso_scores[0,0,:,0]:', torso_scores[0,0,:,0])
        
        sum_torso_scores = torch.sum(torso_scores, 2) #sum over rows of joints
        sum_elbow_scores = torch.sum(elbow_scores, 2) #sum over rows of joints
        sum_wrist_scores = torch.sum(wrist_scores, 2) #sum over rows of joints
        sum_knee_scores = torch.sum(knee_scores, 2) #sum over rows of joints
        sum_ankle_scores = torch.sum(ankle_scores, 2) #sum over rows of joints
        print('sum_torso_scores.shape', sum_torso_scores.shape)
        print('sum_elbow_scores.shape', sum_elbow_scores.shape)
        print('sum_wrist_scores.shape', sum_wrist_scores.shape)
        print('sum_knee_scores.shape', sum_knee_scores.shape)
        print('sum_ankle_scores.shape', sum_ankle_scores.shape)
        #print('sum_torso_scores[0,:,0]:', sum_torso_scores[0,:,0])
        
        sum_torso_scores = torch.sum(sum_torso_scores, 1) #sum over rows of joints
        sum_elbow_scores = torch.sum(sum_elbow_scores, 1) #sum over rows of joints
        sum_wrist_scores = torch.sum(sum_wrist_scores, 1) #sum over rows of joints
        sum_knee_scores = torch.sum(sum_knee_scores, 1) #sum over rows of joints
        sum_ankle_scores = torch.sum(sum_ankle_scores, 1) #sum over rows of joints
        print('SUM sum_torso_scores.shape', sum_torso_scores.shape)
        print('SUM sum_elbow_scores.shape', sum_elbow_scores.shape)
        print('SUM sum_wrist_scores.shape', sum_wrist_scores.shape)
        print('SUM sum_knee_scores.shape', sum_knee_scores.shape)
        print('SUM sum_ankle_scores.shape', sum_ankle_scores.shape)
        print('SUM sum_torso_scores[0]:', sum_torso_scores[0])
        print('SUM sum_elbow_scores[0]:', sum_elbow_scores[0])
        print('SUM sum_wrist_scores[0]:', sum_wrist_scores[0])
        print('SUM sum_knee_scores[0]:', sum_knee_scores[0])
        print('SUM sum_ankle_scores[0]:', sum_ankle_scores[0])
        
        sum_torso_scores_1 = sum_torso_scores[:, 0:7] #joints 0,1,2,3,4,5,6 (head and shoulders)
        sum_torso_scores_2 = sum_torso_scores[:, 7:9] #joints 11,12 (hips)
        print('SUM sum_torso_scores_1.shape', sum_torso_scores_1.shape)
        print('SUM sum_torso_scores_2.shape', sum_torso_scores_2.shape)
        
        sum_body_part_scores = torch.cat((sum_torso_scores_1, sum_elbow_scores, sum_wrist_scores,
                                          sum_torso_scores_2, sum_knee_scores, sum_ankle_scores), dim=1)
        print('sum_body_part_scores.shape', sum_body_part_scores.shape)
        print('sum_body_part_scores[0]:', sum_body_part_scores[0])
        
        
        #visualize_skeleton_and_attention(model_type, test_video, person_id, draw_trajectory_segment, test_frames, scores=sum_scores, spatial_body_part_scores=sum_body_part_scores)
        visualize_skeleton_and_attention(model_type, test_video, person_id, draw_trajectory_segment, test_frames, scores=sum_scores, spatial_scores=sum_body_part_scores)
    elif model_type == 'ttspcc2':
        visualize_skeleton_and_attention(model_type, test_video, person_id, draw_trajectory_segment, test_frames, scores=sum_scores, dataset=dataset)


