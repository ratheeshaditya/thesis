import torch
import torch.nn.functional as F

from trajectory import Trajectory, extract_fixed_sized_segments, split_into_train_and_test, remove_short_trajectories, get_categories, get_UTK_categories
from transformer import TemporalTransformer_4, TemporalTransformer_3, TemporalTransformer_2, BodyPartTransformer, SpatialTemporalTransformer, TemporalTransformer, Block, Attention, Mlp
from transformer_store_attn import TemporalTransformer_store_attn, TemporalTransformer_2_store_attn, TemporalTransformer_3_store_attn, TemporalTransformer_4_store_attn
from transformer_store_attn import SpatialTemporalTransformer_store_attn, BodyPartTransformer_store_attn
from trajectory import Trajectory, extract_fixed_sized_segments, split_into_train_and_test, remove_short_trajectories, get_categories
import pickle
import os
import argparse
import csv
from einops import rearrange

from visualize_attention_weights import visualize_attention_weights
from visualize_skeleton_and_attention import visualize_skeleton_and_attention
#from extract_frames import extract_frames

'''
parser = argparse.ArgumentParser()
parser.add_argument("--filename", help="filename of trained model and testing results")
parser.add_argument("--model_type", help="type of model to train, must be temporal, temporal_2, temporal_3, temporal_4, spatial-temporal or parts", type=str)
parser.add_argument("--embed_dim", help="embedding dimension used by the model", type=int)
parser.add_argument("--segment_length", help="length of sliding window", default=12, type=int)


args = parser.parse_args()

print(args)
'''

#Load test trajectories
PIK_test = "./data/test_anomaly_trajectories.dat"
all_categories = get_categories()

dataset = "HR-Crime"
segment_length = 60
num_classes = 13
num_joints = 17
in_chans = 2
num_parts = 5
device = "cuda:0" if torch.cuda.is_available() else "cpu"

with open(PIK_test, "rb") as f:
    test_crime_trajectories = pickle.load(f)

print('\nLoaded %d test trajectories' % (len(test_crime_trajectories)))

test_crime_trajectories = remove_short_trajectories(test_crime_trajectories, input_length=segment_length, input_gap=0, pred_length=0)
print('\nRemoved short trajectories: %d test trajectories left' % (len(test_crime_trajectories)))

traj_ids_test, traj_videos_test, traj_persons_test, traj_frames_test, traj_categories_test, X_test = extract_fixed_sized_segments(dataset, test_crime_trajectories, input_length=segment_length)
number_of_segments = len(traj_ids_test)
print('number_of_segments', number_of_segments)

test_dataloader = torch.utils.data.DataLoader([ [traj_categories_test[i], traj_videos_test[i], traj_persons_test[i], traj_frames_test[i], X_test[i] ] for i in range(len(traj_ids_test))], shuffle=False, batch_size=number_of_segments) 


# set model parameters for HR-Crime
model_type_1 = 'spatial-temporal'
filename_1 = 'FINAL_spatial_temporal_embed_dim_32_segment_length_60_fold_1'
embed_dim_1 = 32

model_type_2 = 'parts'
filename_2 = 'FIXED_train_body_part_transformer_embed_dim_16_segment_length_60_2_fold_2'
embed_dim_2 = 16

def load_model(model_type, filename, embed_dim):

    PATH = '/data/s3447707/MasterThesis/trained_models/' + filename + '.pt'
    model = torch.load(PATH)
    
    
    NEW_PATH = '/data/s3447707/MasterThesis/saved_state_dict/' + filename + '.pt'
    if not os.path.isfile(NEW_PATH):
        torch.save(model.state_dict(), NEW_PATH)
        print('\nsave model state dict to', NEW_PATH)
    else:
        print('\model state dict %s already exists' % (NEW_PATH))
    
    
    #Create model object
    if model_type == 'temporal':
        model = TemporalTransformer_store_attn(embed_dim=embed_dim, num_frames=segment_length, num_classes=num_classes, num_joints=num_joints, in_chans=in_chans, mlp_ratio=2., qkv_bias=True, qk_scale=None, dropout=0.1)
    elif model_type == 'temporal_2':
        model = TemporalTransformer_2_store_attn(embed_dim=embed_dim, num_frames=segment_length, num_classes=num_classes, num_joints=num_joints, in_chans=in_chans, mlp_ratio=2., qkv_bias=True, qk_scale=None, dropout=0.1)
    elif model_type == 'temporal_3':
        model = TemporalTransformer_3_store_attn(embed_dim=embed_dim, num_frames=segment_length, num_classes=num_classes, num_joints=num_joints, num_parts=num_parts, in_chans=in_chans, mlp_ratio=2., qkv_bias=True, qk_scale=None, dropout=0.1)
    elif model_type == 'temporal_4':
        model = TemporalTransformer_4_store_attn(embed_dim=embed_dim, num_frames=segment_length, num_classes=num_classes, num_joints=num_joints, num_parts=num_parts, in_chans=in_chans, mlp_ratio=2., qkv_bias=True, qk_scale=None, dropout=0.1)
    elif model_type == 'spatial-temporal':
        model = SpatialTemporalTransformer_store_attn(embed_dim_ratio=embed_dim, num_frames=segment_length, num_classes=num_classes, num_joints=num_joints, in_chans=in_chans, mlp_ratio=2., qkv_bias=True, qk_scale=None, dropout=0.1)
    elif model_type == "parts":
        model = BodyPartTransformer_store_attn(embed_dim_ratio=embed_dim, num_frames=segment_length, num_classes=num_classes, num_joints=num_joints, in_chans=in_chans, mlp_ratio=2., qkv_bias=True, qk_scale=None, dropout=0.1)
    else:
        raise Exception('model_type is missing, must be temporal, temporal_2, temporal_3, temporal_4, spatial-temporal or parts')
    
    
    #Load model state dict
    model.load_state_dict(torch.load(NEW_PATH), strict=False)
    return model


model_ST = load_model(model_type_1, filename_1, embed_dim_1)

model_ST.eval()


#Evaluate test samples
def evaluate_model(model, model_type, test_dataloader):
    
    write_dir = '/data/s3447707/MasterThesis/statistical_testing/' + model_type
    
    if not os.path.exists(write_dir):
        # create a subfolder for the model
        print('make new directory ' + str(write_dir))
        os.makedirs(write_dir)
    
    file = os.path.join(write_dir,'paired_t_test.csv')
    f = open(file, "w")
    csv_writer = csv.writer(f, delimiter=';')
    csv_writer.writerow(['label', 'video', 'person', 'prediction'])
    
    all_outputs = torch.tensor([]).to(device)
    all_labels = torch.LongTensor([]).to(device)
    all_videos = torch.LongTensor([]).to(device)
    all_persons = torch.LongTensor([]).to(device)

    for batch in test_dataloader:
        
        labels, videos, persons, frames, data = batch
    
    
        labels = labels.to(device)
        videos = videos.to(device)
        persons = persons.to(device)
        frames = frames.to(device)
        data = data.to(device)
        index = torch.tensor([0]).to(device)
        labels = labels.index_select(1, index)
        labels = torch.squeeze(labels)
        #print('videos',videos)
        videos = videos.index_select(1, index)
        videos = torch.squeeze(videos)
        persons = persons.index_select(1, index)
        persons = torch.squeeze(persons)
        #print('labels length:', len(labels))
        #print('videos:', videos)
        #print(frames)
        #print('frames at index %d: %s' % (test_index, frames[test_index]))
        
        outputs = model(data)
        #print('outputs', outputs)
        
        all_outputs = torch.cat((all_outputs, outputs), 0)
        all_labels = torch.cat((all_labels, labels), 0)
        all_videos = torch.cat((all_videos, videos), 0)
        all_persons = torch.cat((all_persons, persons), 0)
        
        all_log_likelihoods = F.log_softmax(all_outputs, dim=1) #nn.CrossEntropyLoss also uses the log_softmax
        # the class with the highest log-likelihood is what we choose as prediction
        _, all_predictions = torch.max(all_log_likelihoods, dim=1)

        
    for label, video, person, prediction, log_likelihoods, logits in zip(labels, videos, persons, predictions):
        csv_writer.writerow([label.item(),  video.item(), person.item(), prediction.item()])
    
    f.close()


evaluate_model(model_ST, model_type_1, test_dataloader)