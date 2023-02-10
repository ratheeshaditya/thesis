# %%
'''
Script to obtain embeddings for T-SNE and Silhouette plots
'''

import torch


import torch.nn as nn
from einops import rearrange, repeat
import torch.nn.functional as F

import numpy as np

from trajectory import Trajectory, extract_fixed_sized_segments, split_into_train_and_test, remove_short_trajectories, get_categories, get_UTK_categories
# from transformer_store_attn import TubeletTemporalSpatialPart_concat_chan_2_Transformer_store_attn, TubeletTemporalPart_concat_chan_1_Transformer_store_attn, TubeletTemporalTransformer_store_attn, TubeletTemporalPart_mean_chan_1_Transformer_store_attn, TubeletTemporalPart_mean_chan_2_Transformer_store_attn, TubeletTemporalPart_concat_chan_2_Transformer_store_attn, TemporalTransformer_4_store_attn, TemporalTransformer_3_store_attn, TemporalTransformer_2_store_attn, BodyPartTransformer_store_attn, SpatialTemporalTransformer_store_attn, TemporalTransformer_store_attn
from trajectory import TrajectoryDataset, Trajectory, extract_fixed_sized_segments, split_into_train_and_test, remove_short_trajectories, get_NTU_categories, get_categories
import pickle
import os
import argparse
import yaml
from utils import print_statistics, SetupLogger, evaluate_all, evaluate_category, conv_to_float, SetupFolders, train_acc, SetupVisFolders


# %%
# model_type = 'ttspcc2'
model_type = 'spatial-temporal'
filename = 'HRC_st_tsne_plotting'
# filename = 'HRC_ttspcc2_233'
embed_dim = 32
test_sample = 1
test_index = 1
segment_length = 60
# dataset = 'NTU_2D'
dataset = 'HRC'
vis_type = 1
kernel = '8,3,3'
# kernel = '2,3,3'
stride = kernel
dec = False
device = 'cuda'

# %%
def get_average_body_parts(num_joints, x):
    if num_joints == 25:
        dim = int(x.size(2)/25)
        x_torso_1 = x[:, :, 0:5*dim]
        x_torso_2 = x[:, :, 8*dim:9*dim]
        x_torso_3 = x[:, :, 12*dim:13*dim]
        x_torso_4 = x[:, :, 16*dim:17*dim]
        x_torso_5 = x[:, :, 20*dim:21*dim]
        x_torso = torch.cat((x_torso_1, x_torso_2, x_torso_3, x_torso_4, x_torso_5), dim=2)

        x_wrist_1 = x[:, :, 6*dim:7*dim]
        x_wrist_2 = x[:, :, 7*dim:8*dim]
        x_wrist_3 = x[:, :, 10*dim:11*dim]
        x_wrist_4 = x[:, :, 11*dim:12*dim]
        x_wrist_5 = x[:, :, 21*dim:22*dim]
        x_wrist_6 = x[:, :, 22*dim:23*dim]
        x_wrist_7 = x[:, :, 23*dim:24*dim]
        x_wrist_8 = x[:, :, 24*dim:25*dim]
        x_wrist = torch.cat((x_wrist_1, x_wrist_2, x_wrist_3, x_wrist_4, x_wrist_5, x_wrist_6, x_wrist_7, x_wrist_8), dim=2)

        x_elbow_1 = x[:, :, 9*dim:10*dim]
        x_elbow_2 = x[:, :, 5*dim:6*dim]
        x_elbow = torch.cat((x_elbow_1, x_elbow_2), dim=2)

        x_knee_1 = x[:, :, 17*dim:18*dim]
        x_knee_2 = x[:, :, 13*dim:14*dim]
        x_knee = torch.cat((x_knee_1, x_knee_2), dim=2)

        x_ankle_1 = x[:, :, 18*dim:19*dim]
        x_ankle_2 = x[:, :, 19*dim:20*dim]
        x_ankle_3 = x[:, :, 14*dim:15*dim]
        x_ankle_4 = x[:, :, 15*dim:16*dim]
        x_ankle = torch.cat((x_ankle_1, x_ankle_2, x_ankle_3, x_ankle_4), dim=2)

        x_torso_x = x_torso[:, :, ::2]
        x_elbow_x = x_elbow[:, :, ::2]
        x_wrist_x = x_wrist[:, :, ::2]
        x_knee_x = x_knee[:, :, ::2]
        x_ankle_x = x_ankle[:, :, ::2]

        x_torso_y = x_torso[:, :, 1::2]
        x_elbow_y = x_elbow[:, :, 1::2]
        x_wrist_y = x_wrist[:, :, 1::2]
        x_knee_y = x_knee[:, :, 1::2]
        x_ankle_y = x_ankle[:, :, 1::2]

        x_torso_x = torch.mean(torch.Tensor.float(x_torso_x), dim=2)
        x_elbow_x = torch.mean(torch.Tensor.float(x_elbow_x), dim=2)
        x_wrist_x = torch.mean(torch.Tensor.float(x_wrist_x), dim=2)
        x_knee_x = torch.mean(torch.Tensor.float(x_knee_x), dim=2)
        x_ankle_x = torch.mean(torch.Tensor.float(x_ankle_x), dim=2)

        x_torso_y = torch.mean(torch.Tensor.float(x_torso_y), dim=2)
        x_elbow_y = torch.mean(torch.Tensor.float(x_elbow_y), dim=2)
        x_wrist_y = torch.mean(torch.Tensor.float(x_wrist_y), dim=2)
        x_knee_y = torch.mean(torch.Tensor.float(x_knee_y), dim=2)
        x_ankle_y = torch.mean(torch.Tensor.float(x_ankle_y), dim=2)

        x_torso_x = torch.unsqueeze(x_torso_x, 2)
        x_elbow_x = torch.unsqueeze(x_elbow_x, 2)
        x_wrist_x = torch.unsqueeze(x_wrist_x, 2)
        x_knee_x = torch.unsqueeze(x_knee_x, 2)
        x_ankle_x = torch.unsqueeze(x_ankle_x, 2)

        x_torso_y = torch.unsqueeze(x_torso_y, 2)
        x_elbow_y = torch.unsqueeze(x_elbow_y, 2)
        x_wrist_y = torch.unsqueeze(x_wrist_y, 2)
        x_knee_y = torch.unsqueeze(x_knee_y, 2)
        x_ankle_y = torch.unsqueeze(x_ankle_y, 2)

        x_torso = torch.cat((x_torso_x, x_torso_y), dim=2)
        x_elbow = torch.cat((x_elbow_x, x_elbow_y), dim=2)
        x_wrist = torch.cat((x_wrist_x, x_wrist_y), dim=2)
        x_knee = torch.cat((x_knee_x, x_knee_y), dim=2)
        x_ankle = torch.cat((x_ankle_x, x_ankle_y), dim=2)

        x = torch.cat((x_torso, x_elbow, x_wrist, x_knee, x_ankle), dim=2)
        return x


    elif num_joints == 17:
        #x_torso = x[:, :, 0:9*2]
        x_torso_1 = x[:, :, 0:7*2] #joints 0,1,2,3,4,5,6 (head and shoulders) 
        x_torso_2 = x[:, :, 11*2:13*2] #joints 11,12 (hips)
        #print('x_torso_1[0]', x_torso_1[0])
        #print('x_torso_2[0]', x_torso_2[0])
        x_torso = torch.cat((x_torso_1, x_torso_2), dim=2)
        #print('x_torso[0]', x_torso[0])
        
        x_elbow = x[:, :, 7*2:9*2]
        x_wrist = x[:, :, 9*2:11*2]
        x_knee = x[:, :, 13*2:15*2]
        x_ankle = x[:, :, 15*2:17*2]

        '''
        print('x_torso shape', x_torso.shape)
        print('x_elbow shape', x_elbow.shape)
        print('x_wrist shape', x_wrist.shape)
        print('x_knee shape', x_knee.shape)
        print('x_ankle shape', x_ankle.shape)
        '''

        x_torso_x = x_torso[:, :, ::2]
        x_elbow_x = x_elbow[:, :, ::2]
        x_wrist_x = x_wrist[:, :, ::2]
        x_knee_x = x_knee[:, :, ::2]
        x_ankle_x = x_ankle[:, :, ::2]

        '''
        print('\nx_torso_x shape', x_torso_x.shape)
        print('x_elbow_x shape', x_elbow_x.shape)
        print('x_wrist_x shape', x_wrist_x.shape)
        print('x_knee_x shape', x_knee_x.shape)
        print('x_ankle_x shape', x_ankle_x.shape)
        '''

        x_torso_y = x_torso[:, :, 1::2]
        x_elbow_y = x_elbow[:, :, 1::2]
        x_wrist_y = x_wrist[:, :, 1::2]
        x_knee_y = x_knee[:, :, 1::2]
        x_ankle_y = x_ankle[:, :, 1::2]

        #print('\nx_torso_x', x_torso_x)
        #print('x_torso_y', x_torso_y)

        x_torso_x = torch.mean(torch.Tensor.float(x_torso_x), dim=2)
        x_elbow_x = torch.mean(torch.Tensor.float(x_elbow_x), dim=2)
        x_wrist_x = torch.mean(torch.Tensor.float(x_wrist_x), dim=2)
        x_knee_x = torch.mean(torch.Tensor.float(x_knee_x), dim=2)
        x_ankle_x = torch.mean(torch.Tensor.float(x_ankle_x), dim=2)

        x_torso_y = torch.mean(torch.Tensor.float(x_torso_y), dim=2)
        x_elbow_y = torch.mean(torch.Tensor.float(x_elbow_y), dim=2)
        x_wrist_y = torch.mean(torch.Tensor.float(x_wrist_y), dim=2)
        x_knee_y = torch.mean(torch.Tensor.float(x_knee_y), dim=2)
        x_ankle_y = torch.mean(torch.Tensor.float(x_ankle_y), dim=2)

        x_torso_x = torch.unsqueeze(x_torso_x, 2)
        x_elbow_x = torch.unsqueeze(x_elbow_x, 2)
        x_wrist_x = torch.unsqueeze(x_wrist_x, 2)
        x_knee_x = torch.unsqueeze(x_knee_x, 2)
        x_ankle_x = torch.unsqueeze(x_ankle_x, 2)

        x_torso_y = torch.unsqueeze(x_torso_y, 2)
        x_elbow_y = torch.unsqueeze(x_elbow_y, 2)
        x_wrist_y = torch.unsqueeze(x_wrist_y, 2)
        x_knee_y = torch.unsqueeze(x_knee_y, 2)
        x_ankle_y = torch.unsqueeze(x_ankle_y, 2)

        '''
        print('\nx_torso_x shape', x_torso_x.shape)
        print('x_elbow_x shape', x_elbow_x.shape)
        print('x_wrist_x shape', x_wrist_x.shape)
        print('x_knee_x shape', x_knee_x.shape)
        print('x_ankle_x shape', x_ankle_x.shape)

        print('\nx_torso_x', x_torso_x)
        print('x_torso_y', x_torso_y)
        '''


        x_torso = torch.cat((x_torso_x, x_torso_y), dim=2)
        x_elbow = torch.cat((x_elbow_x, x_elbow_y), dim=2)
        x_wrist = torch.cat((x_wrist_x, x_wrist_y), dim=2)
        x_knee = torch.cat((x_knee_x, x_knee_y), dim=2)
        x_ankle = torch.cat((x_ankle_x, x_ankle_y), dim=2)

        '''
        print('\nx_torso shape', x_torso.shape)
        print('x_elbow shape', x_elbow.shape)
        print('x_wrist shape', x_wrist.shape)
        print('x_knee shape', x_knee.shape)
        print('x_ankle shape', x_ankle.shape)

        print('\nx_torso', x_torso)
        print('\nx_ankle', x_ankle)
        '''

        x = torch.cat((x_torso, x_elbow, x_wrist, x_knee, x_ankle), dim=2)
        return x


def get_keypoint(skeleton, position, dim):
    # device = torch.device('cuda')# if torch.cuda.is_available() else 'cpu')
    # part = torch.empty(0)
    # part.to(device)
    # part = None
    for joint in position:
        x = skeleton[:, :, (joint-1)*dim : joint*dim]
        # print(x.device)
        # print(part.device)
        try:
            part = torch.cat((part, x), dim=2)
        except:
            part = x
        # if position.index(x) == 0:
        #     part = x
        # else:
        #     part = torch.cat((part, x), dim=2)
    return part
#Transformer model
class Mlp(nn.Module):
    """ MLP as used in Vision Transformer, MLP-Mixer and related networks
    """
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x
    
class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        
        # NOTE scale factor can be manually set to be compat with prev weights
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]   # make torchscript happy (cannot use tensor as tuple)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class Block(nn.Module):

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 dropout=0., act_layer=nn.GELU):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim, eps=1e-6)
        self.attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        #self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.dropout = nn.Dropout(dropout) #first try a simple dropout instead of drop path
        self.norm2 = nn.LayerNorm(dim, eps=1e-6)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x):
        #x = x + self.drop_path(self.attn(self.norm1(x)))
        #x = x + self.drop_path(self.mlp(self.norm2(x)))
        x = x + self.dropout(self.attn(self.norm1(x)))
        x = x + self.dropout(self.mlp(self.norm2(x)))
        return x

class TubeletTemporalSpatialPart_concat_chan_2_Transformer(nn.Module):
    def __init__(self, dataset=None, num_classes=13, num_frames=12, num_joints=17, in_chans=2, embed_dim_ratio=64, kernel=None, stride=None, depth=4,
                 num_heads=8, mlp_ratio=2., qkv_bias=True, qk_scale=None,
                 drop_rate=0., attn_drop_rate=0., dropout=0.2, pad_mode='constant'):
        """    ##########hybrid_backbone=None, representation_size=None,
        Args:
            num_classes (int): number of classes for classification head, HR-Crime constists of 13 crime categories
            num_frames (int): number of input frames
            num_joints (int): number of joints per skeleton
            in_chans (int): number of input channels, 2D joints have 2 channels: (x,y)
            embed_dim_ratio (int): embedding dimension ratio
            depth (int): depth of transformer
            num_heads (int): number of attention heads
            mlp_ratio (int): ratio of mlp hidden dim to embedding dim
            qkv_bias (bool): enable bias for qkv if True
            qk_scale (float): override default qk scale of head_dim ** -0.5 if set
            drop_rate (float): dropout rate
            attn_drop_rate (float): attention dropout rate
            drop_path_rate (float): stochastic depth rate
        """
        super().__init__()

        self.in_chans = in_chans
        self.dataset = dataset
        self.pad_mode = pad_mode
        self.num_embed = (int((num_frames-kernel[0])/stride[0]) + 1) * (int((3-kernel[1])/stride[1]) + 1) * (int((3-kernel[2])/stride[2]) + 1)
        
        embed_dim = embed_dim_ratio * self.num_embed #* 5   #### temporal embed_dim is embed_dim_ratio x num_embed (op of 3dconv) x 5 (no. of body parts)

        ### Tubelet Embedder
        if "NTU" in dataset or "HRC" in dataset:
            self.torso_conv = torch.nn.Conv3d(in_chans, embed_dim_ratio, kernel_size=kernel, stride=stride)
            self.elbows_conv = torch.nn.Conv3d(in_chans, embed_dim_ratio, kernel_size=kernel, stride=stride)
            self.wrists_conv = torch.nn.Conv3d(in_chans, embed_dim_ratio, kernel_size=kernel, stride=stride)
            self.knees_conv = torch.nn.Conv3d(in_chans, embed_dim_ratio, kernel_size=kernel, stride=stride)
            self.ankles_conv = torch.nn.Conv3d(in_chans, embed_dim_ratio, kernel_size=kernel, stride=stride)

        ### Tubelet Embedder - Position embedding
        if self.dataset == "HRC":
            self.Torso_pos_embed = nn.Parameter(torch.zeros(self.num_embed, embed_dim_ratio)) #9 joints
            self.Elbow_pos_embed = nn.Parameter(torch.zeros(self.num_embed, embed_dim_ratio)) #2 joints in remaining body parts
            self.Wrist_pos_embed = nn.Parameter(torch.zeros(self.num_embed, embed_dim_ratio)) #2 joints in remaining body parts
            self.Knee_pos_embed = nn.Parameter(torch.zeros(self.num_embed, embed_dim_ratio)) #2 joints in remaining body parts
            self.Ankle_pos_embed = nn.Parameter(torch.zeros(self.num_embed, embed_dim_ratio)) #2 joints in remaining body parts
        elif "NTU" in self.dataset:
            if "2D" in self.dataset:
                self.Torso_pos_embed = nn.Parameter(torch.zeros(self.num_embed, embed_dim_ratio)) #9 joints
                self.Elbow_pos_embed = nn.Parameter(torch.zeros(self.num_embed, embed_dim_ratio)) #2 joints in remaining body parts 
                self.Wrist_pos_embed = nn.Parameter(torch.zeros(self.num_embed, embed_dim_ratio)) #2 joints in remaining body parts 
                self.Knee_pos_embed = nn.Parameter(torch.zeros(self.num_embed, embed_dim_ratio)) #2 joints in remaining body parts 
                self.Ankle_pos_embed = nn.Parameter(torch.zeros(self.num_embed, embed_dim_ratio)) #2 joints in remaining body parts 


        self.Temporal_pos_embed = nn.Parameter(torch.zeros(5 + 1, embed_dim))
        self.pos_drop = nn.Dropout(p=drop_rate)

        '''
        self.Spatial_blocks = nn.ModuleList([
            Block(
                dim=embed_dim_ratio, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, dropout=dropout)
            for i in range(depth)])
        '''

        # Spatial
        
        self.Torso_blocks = nn.ModuleList([
            Block(
                dim=embed_dim_ratio, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, dropout=dropout)
            for i in range(depth)])
        
        self.Elbow_blocks = nn.ModuleList([
            Block(
                dim=embed_dim_ratio, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, dropout=dropout)
            for i in range(depth)])
        
        self.Wrist_blocks = nn.ModuleList([
            Block(
                dim=embed_dim_ratio, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, dropout=dropout)
            for i in range(depth)])

        self.Knee_blocks = nn.ModuleList([
            Block(
                dim=embed_dim_ratio, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, dropout=dropout)
            for i in range(depth)])

        self.Ankle_blocks = nn.ModuleList([
            Block(
                dim=embed_dim_ratio, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, dropout=dropout)
            for i in range(depth)])
        
        # Temporal
        self.blocks = nn.ModuleList([
            Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, dropout=dropout)
            for i in range(depth)])

        #self.norm = nn.LayerNorm(embed_dim, eps=1e-6)
        self.Spatial_norm =  nn.LayerNorm(embed_dim_ratio, eps=1e-6)
        self.Temporal_norm =  nn.LayerNorm(embed_dim, eps=1e-6)

        print('num_classes',num_classes)
        print('embed_dim', embed_dim)

        self.cls_token = nn.Parameter(torch.zeros(1, embed_dim))
        

        # Classifier head(s)
        "Define standard linear to map the final output sequence to class logits"
        self.head = nn.Linear(embed_dim, num_classes) #do not use softmax here. nn.CrossEntropyLoss takes the logits as input and calculates the softmax
        
        #print('self.head',self.head)
        #print('num_classes',num_classes)

        # initialize weights
        self.init_weights()

        # taken from https://pytorch.org/tutorials/beginner/transformer_tutorial.html
    def init_weights(self):
        initrange = 0.1
    #   self.Spatial_patch_to_embedding.weight.data.uniform_(-initrange, initrange)
        self.torso_conv.weight.data.uniform_(-initrange, initrange)
        self.elbows_conv.weight.data.uniform_(-initrange, initrange)
        self.wrists_conv.weight.data.uniform_(-initrange, initrange)
        self.knees_conv.weight.data.uniform_(-initrange, initrange)
        self.ankles_conv.weight.data.uniform_(-initrange, initrange)

        self.head.bias.data.zero_()
        self.head.weight.data.uniform_(-initrange, initrange)
    
    def Torso_forward_features(self, x):
        x = self.torso_conv(x)
        x = rearrange(x, "b e n1 n2 n3 -> b (n1 n2 n3) e")
        x = self.pos_drop(x + self.Torso_pos_embed)

        for blk in self.Torso_blocks:
            x = blk(x)
        
        x = self.Spatial_norm(x)
        x = rearrange(x, "b n e -> b (n e)")

        return x
    
    def Elbow_forward_features(self, x):
        x = self.elbows_conv(x)
        x = rearrange(x, "b e n1 n2 n3 -> b (n1 n2 n3) e")
        x = self.pos_drop(x + self.Elbow_pos_embed)

        for blk in self.Elbow_blocks:
            x = blk(x)
        
        x = self.Spatial_norm(x)
        x = rearrange(x, "b n e -> b (n e)")

        return x
    
    def Wrist_forward_features(self, x):
        x = self.wrists_conv(x)
        x = rearrange(x, "b e n1 n2 n3 -> b (n1 n2 n3) e")
        x = self.pos_drop(x + self.Wrist_pos_embed)

        for blk in self.Wrist_blocks:
            x = blk(x)
        
        x = self.Spatial_norm(x)
        x = rearrange(x, "b n e -> b (n e)")

        return x
    
    def Knee_forward_features(self, x):
        x = self.knees_conv(x)
        x = rearrange(x, "b e n1 n2 n3 -> b (n1 n2 n3) e")
        x = self.pos_drop(x + self.Knee_pos_embed)

        for blk in self.Knee_blocks:
            x = blk(x)
        
        x = self.Spatial_norm(x)
        x = rearrange(x, "b n e -> b (n e)")

        return x

    def Ankle_forward_features(self, x):
        x = self.ankles_conv(x)
        x = rearrange(x, "b e n1 n2 n3 -> b (n1 n2 n3) e")
        x = self.pos_drop(x + self.Ankle_pos_embed)

        for blk in self.Ankle_blocks:
            x = blk(x)
        
        x = self.Spatial_norm(x)
        x = rearrange(x, "b n e -> b (n e)")

        return x

    def forward_features(self, x):
        cls_token = self.cls_token.expand(x.shape[0], -1, -1)  # stole cls_tokens impl from Phil Wang, thanks

        x = torch.cat((cls_token, x), dim=1)

        x = self.pos_drop(x + self.Temporal_pos_embed)

        for blk in self.blocks:
            x = blk(x)

        x = self.Temporal_norm(x)
        cls_token_final = x[:, 0]
        return cls_token_final


    def tubelet_embedding(self, x):
        if self.dataset == "NTU_3D":
            torso = get_keypoint(x, [], 3)
        elif self.dataset == "NTU_2D":
            torso = get_keypoint(x, [4, 3, 9, 21, 5, 2, 17, 1, 13], 2)
            elbows = get_keypoint(x, [10, 6], 2)
            wrists = get_keypoint(x, [11, 12, 24, 25, 7, 8, 22, 23], 2)
            knees = get_keypoint(x, [18, 14], 2)
            ankles = get_keypoint(x, [19, 20, 15, 16], 2)

            torso = rearrange(torso, "b f (x y c) -> b c f x y", x= 3, y =3)       ## shape: b 2 f 3 3
            # torso = F.pad(input=torso, pad=(0, 0, 2, 1), mode='constant', value=0)   ## Shape: b f 6 6 
            # torso = torso.unsqueeze(1)                                         ## Shape: b 1 f 6 6

            elbows = rearrange(elbows, "b f (x y c) -> b c f x y", x= 1, y =2)     ## Shape: b f 1 2 
            if self.pad_mode == 'constant':
                elbows = F.pad(input=elbows, pad=(1, 0, 1, 1), mode='constant', value=0) ## Shape: b f 2 2
            else:
                elbows = F.pad(input=elbows, pad=(1, 0, 1, 1, 0, 0), mode=self.pad_mode) ## Shape: b f 2 2
            # elbows = elbows.unsqueeze(1)

            wrists = F.pad(wrists ,(0,2)) # Extend wrists by one keypoint so that we can transform it into 3x3 
            wrists = rearrange(wrists, "b f (x y c) -> b c f x y", x= 3, y =3)     ## Shape: b f 3 3
            # wrists = F.pad(input=wrists, pad=(0, 0, 1, 1), mode='constant', value=0) ## Shape: b f 4 4
            # wrists = wrists.unsqueeze(1)                                       ## Shape: b 1 f 6 6

            knees = rearrange(knees, "b f (x y c) -> b c f x y", x= 1, y =2)       ## Shape: b f 1 2
            if self.pad_mode == 'constant':
                knees = F.pad(input=knees, pad=(1, 0, 1, 1), mode='constant', value=0) ## Shape: b f 2 2
            else:
                knees = F.pad(input=knees, pad=(1, 0, 1, 1, 0, 0), mode=self.pad_mode) ## Shape: b f 2 2
            # knees = knees.unsqueeze(1)                                         ## Shape: b 1 f 6 6

            ankles = rearrange(ankles, "b f (x y c) -> b c f x y", x= 2, y =2)     ## Shape: b f 2 2
            if self.pad_mode == 'constant':
                ankles = F.pad(input=ankles, pad=(1, 0, 1, 0), mode='constant', value=0) ## Shape: b f 6 6
            else:
                ankles = F.pad(input=ankles, pad=(1, 0, 1, 0, 0, 0), mode=self.pad_mode) ## Shape: b f 6 6
            # ankles = ankles.unsqueeze(1)                                       ## Shape: b 1 f 6 6

        elif self.dataset == "HRC":
            torso = get_keypoint(x, [1, 2, 3, 4, 5, 6, 7, 12, 13], 2)
            elbows = get_keypoint(x, [8, 9], 2)
            wrists = get_keypoint(x, [10, 11], 2)
            knees = get_keypoint(x, [14, 15], 2)
            ankles = get_keypoint(x, [16, 17], 2)

            torso = rearrange(torso, "b f (x y c) -> b c f x y", x= 3, y =3)       ## shape: b 2 f 3 3
            # torso = F.pad(input=torso, pad=(0, 0, 2, 1), mode='constant', value=0)   ## Shape: b f 6 6 
            # torso = torso.unsqueeze(1)                                         ## Shape: b 1 f 6 6

            elbows = rearrange(elbows, "b f (x y c) -> b c f x y", x= 1, y =2)     ## Shape: b 2 f 1 2 
            if self.pad_mode == 'constant':
                elbows = F.pad(input=elbows, pad=(1, 0, 1, 1), mode='constant', value=0) ## Shape: b f 6 6
            else:
                elbows = F.pad(input=elbows, pad=(1, 0, 1, 1, 0, 0), mode=self.pad_mode) ## Shape: b f 6 6
            # elbows = elbows.unsqueeze(1)

            wrists = rearrange(wrists, "b f (x y c) -> b c f x y", x= 1, y =2)     ## Shape: b 2 f 1 2
            if self.pad_mode == 'constant':
                wrists = F.pad(input=wrists, pad=(1, 0, 1, 1), mode='constant', value=0) ## Shape: b f 6 6
            else:
                wrists = F.pad(input=wrists, pad=(1, 0, 1, 1, 0, 0), mode=self.pad_mode) ## Shape: b f 6 6
            # wrists = wrists.unsqueeze(1)                                       ## Shape: b 1 f 6 6

            knees = rearrange(knees, "b f (x y c) -> b c f x y", x= 1, y =2)       ## Shape: b 2 f 1 2
            if self.pad_mode == 'constant':
                knees = F.pad(input=knees, pad=(1, 0, 1, 1), mode='constant', value=0) ## Shape: b f 6 6
            else:
                knees = F.pad(input=knees, pad=(1, 0, 1, 1, 0, 0), mode=self.pad_mode) ## Shape: b f 6 6
            # knees = knees.unsqueeze(1)                                         ## Shape: b 1 f 6 6

            ankles = rearrange(ankles, "b f (x y c) -> b c f x y", x= 1, y =2)     ## Shape: b 2 f 1 2
            if self.pad_mode == 'constant':
                ankles = F.pad(input=ankles, pad=(1, 0, 1, 1), mode='constant', value=0) ## Shape: b f 6 6
            else:
                ankles = F.pad(input=ankles, pad=(1, 0, 1, 1, 0, 0), mode=self.pad_mode) ## Shape: b f 6 6
            # ankles = ankles.unsqueeze(1)                                       ## Shape: b 1 f 6 6

        torso_embed = self.Torso_forward_features(torso)
        elbows_embed = self.Elbow_forward_features(elbows)
        wrists_embed = self.Wrist_forward_features(wrists)
        knees_embed = self.Knee_forward_features(knees)
        ankles_embed = self.Ankle_forward_features(ankles)

        x = torch.stack((torso_embed, elbows_embed, wrists_embed, knees_embed, ankles_embed), dim=1)

        return x

    def forward(self, x):
        x = self.tubelet_embedding(x)
     
        x = self.forward_features(x)
        
        # x = self.head(x)
        # x = F.log_softmax(x, dim=1)
        return x

class SpatialTemporalTransformer(nn.Module):
    def __init__(self, num_classes=13, num_frames=12, num_joints=17, in_chans=2, embed_dim_ratio=8, depth=4,
                 num_heads=8, mlp_ratio=2., qkv_bias=True, qk_scale=None,
                 drop_rate=0., attn_drop_rate=0., dropout=0.2):
        """    ##########hybrid_backbone=None, representation_size=None,
        Args:
            num_classes (int): number of classes for classification head, HR-Crime constists of 13 crime categories
            num_frames (int): number of input frames
            num_joints (int): number of joints per skeleton
            in_chans (int): number of input channels, 2D joints have 2 channels: (x,y)
            embed_dim_ratio (int): embedding dimension ratio
            depth (int): depth of transformer
            num_heads (int): number of attention heads
            mlp_ratio (int): ratio of mlp hidden dim to embedding dim
            qkv_bias (bool): enable bias for qkv if True
            qk_scale (float): override default qk scale of head_dim ** -0.5 if set
            drop_rate (float): dropout rate
            attn_drop_rate (float): attention dropout rate
            drop_path_rate (float): stochastic depth rate
        """
        super().__init__()
        
        print('num_classes',num_classes)
        print('embed_dim_ratio', embed_dim_ratio)
        print('in_chans', in_chans)
        print('num_joints', num_joints)
        
        self.in_chans = in_chans

        embed_dim = embed_dim_ratio * num_joints   #### temporal embed_dim is num_joints * spatial embedding dim ratio

        #self.embedding = nn.Linear(num_joints*2, embed_dim)
        #self.pos_embed = nn.Parameter(torch.zeros(num_frames+1, embed_dim))

        ### spatial patch embedding
        self.Spatial_patch_to_embedding = nn.Linear(in_chans, embed_dim_ratio)
        self.Spatial_pos_embed = nn.Parameter(torch.zeros(num_joints, embed_dim_ratio))

        self.Temporal_pos_embed = nn.Parameter(torch.zeros(num_frames + 1, embed_dim))
        self.pos_drop = nn.Dropout(p=drop_rate)

        self.Spatial_blocks = nn.ModuleList([
            Block(
                dim=embed_dim_ratio, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, dropout=dropout)
            for i in range(depth)])

        self.blocks = nn.ModuleList([
            Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, dropout=dropout)
            for i in range(depth)])

        #self.norm = nn.LayerNorm(embed_dim, eps=1e-6)
        self.Spatial_norm =  nn.LayerNorm(embed_dim_ratio, eps=1e-6)
        self.Temporal_norm =  nn.LayerNorm(embed_dim, eps=1e-6)

        print('num_classes',num_classes)
        print('embed_dim', embed_dim)

        self.cls_token = nn.Parameter(torch.zeros(1, embed_dim))
        

        # Classifier head(s)
        "Define standard linear to map the final output sequence to class logits"
        self.head = nn.Linear(embed_dim, num_classes) #do not use softmax here. nn.CrossEntropyLoss takes the logits as input and calculates the softmax
        
        #print('self.head',self.head)
        #print('num_classes',num_classes)

        # initialize weights
        self.init_weights()

        # taken from https://pytorch.org/tutorials/beginner/transformer_tutorial.html
    def init_weights(self):
          initrange = 0.1
          self.Spatial_patch_to_embedding.weight.data.uniform_(-initrange, initrange)
          self.head.bias.data.zero_()
          self.head.weight.data.uniform_(-initrange, initrange)

    
    def Spatial_forward_features(self, x):
        #print('\nCall Spatial_forward_features')
        #print('x.shape', x.shape)

        b, f, p, c = x.shape  ##### b is batch size, f is number of frames, p is number of joints
        x = rearrange(x, 'b f p c -> (b f) p c', ) ####concatenate coordinates along frames and reorder axes

        #print('new x.shape', x.shape)
        #print('new x', x)

        x = self.Spatial_patch_to_embedding(x)
        x = self.pos_drop(x + self.Spatial_pos_embed)

        for blk in self.Spatial_blocks:
            x = blk(x)

        x = self.Spatial_norm(x)
        x = rearrange(x, '(b f) w c -> b f (w c)', f=f) ####rearrange tensor to match temporal transformer input shape [batch_size, num_frames, embed_dim]

        #print('rearranged x.shape', x.shape)
        return x
    
    
    def Spatial_forward_features_mistake(self, x):
        #print('\nCall Spatial_forward_features')
        #print('x.shape', x.shape)
        b, _, f, p = x.shape  ##### b is batch size, f is number of frames, p is number of joints
        x = rearrange(x, 'b c f p  -> (b f) p  c', ) ####concatenate coordinates along frames and reorder axes

        #print('new x.shape', x.shape)

        x = self.Spatial_patch_to_embedding(x)
        x = self.pos_drop(x + self.Spatial_pos_embed)

        for blk in self.Spatial_blocks:
            x = blk(x)

        x = self.Spatial_norm(x)
        x = rearrange(x, '(b f) w c -> b f (w c)', f=f) ####rearrange tensor to match temporal transformer input shape [batch_size, num_frames, embed_dim]

        #print('rearranged x.shape', x.shape)
        return x

    def forward_features(self, x):
        #print('\nCall forward_features')
        #print('x.shape[0]', x.shape[0])
        b  = x.shape[0]

        #print(f"self cls_token shape: {self.cls_token.shape}")

        #print("expand cls_token")
        cls_token = self.cls_token.expand(x.shape[0], -1, -1)  # stole cls_tokens impl from Phil Wang, thanks

        #print(f"expanded cls_token shape: {cls_token.shape}")
        #print(f"spatial encoded x shape: {x.shape}")

        x = torch.cat((cls_token, x), dim=1)

        #print(f"spatial encoded x + cls_token shape: {x.shape}")

        #print(f"Temporal_pos_embed shape: {self.Temporal_pos_embed.shape}")
        #print(f"x + self.Temporal_pos_embed shape: {(x + self.Temporal_pos_embed).shape}")

        x = self.pos_drop(x + self.Temporal_pos_embed)

        for blk in self.blocks:
            x = blk(x)

        x = self.Temporal_norm(x)
        ##### x size [b, f, emb_dim], then take weighted mean on frame dimension, we only predict 3D pose of the center frame
        
        cls_token_final = x[:, 0]
        #print(f"cls_token_final shape: {cls_token_final.shape}")
        #return self.pre_logits(x[:, 0])
        return cls_token_final
    
    
    def forward(self, x):
        #print('\nCall forward')
        #print('x.shape', x.shape)
        #print('x', x)
        b, f, e = x.shape  ##### b is batch size, f is number of frames, e is number of elements equal to 2xnumber of joints
        c = self.in_chans ##### number of channels, in our case 2
        #print('b %d, f %d, e %d' %(b,f,e))
        #print('c',c)
        j = e//c ##### number of joints
        #print('j',j)

        x = torch.reshape(x, (b, f, j, c))
        #print('x.shape', x.shape)
        #print('x reshape', x)

        ### now x is [batch_size, 2 channels, receptive frames, joint_num], following image data
        x = self.Spatial_forward_features(x)
        x = self.forward_features(x)
        
        # x = self.head(x)
        # x = F.log_softmax(x, dim=1)


        return x

    def forward_mistake(self, x):
        #print('\nCall forward')
        #print('x.shape', x.shape)
        b, f, e = x.shape  ##### b is batch size, f is number of frames, e is number of elements equal to 2xnumber of joints
        c = self.in_chans ##### number of channels, in our case 2
        #print('b %d, f %d, e %d' %(b,f,e))
        #print('c',c)
        j = e//c ##### number of joints
        #print('j',j)
        x = x.view(b, c, f, j)
        #print('x.shape', x.shape)
        #x = x.permute(0, 3, 1, 2)
        #b, _, _, p = x.shape
        ### now x is [batch_size, 2 channels, receptive frames, joint_num], following image data
        x = self.Spatial_forward_features(x)
        x = self.forward_features(x)
        
        x = self.head(x)
        #x = x.view(b, 1, p, -1)

        #print(f"head(x) size: {x.size()}")

        return x
# %%
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

# %%
PATH = os.path.join('/home/s2435462/HRC/results/', dataset, filename, 'models', filename+'_fold_1.pt')
# PATH = '/home/s2435462/HRC/results/'+dataset+'/NTU_2D_ttpcc1/models'
# PATH = '/data/s3447707/MasterThesis/trained_models/' + filename + '.pt'
model_ = torch.load(PATH)


if not os.path.exists('/home/s2435462/HRC/results/tsne_silhouette' +'/'+ filename):
    os.mkdir('/home/s2435462/HRC/results/tsne_silhouette' +'/'+ filename)
NEW_PATH = '/home/s2435462/HRC/results/tsne_silhouette' +'/'+ filename + '/state_dict.pt'
if not os.path.isfile(NEW_PATH):
    torch.save(model_.state_dict(), NEW_PATH)
    print('\nsave model state dict to', NEW_PATH)
else:
    print('\model state dict %s already exists' % (NEW_PATH))

# %%
if model_type == 'ttspcc2':
    kernel = tuple(map(int, kernel.split(',')))
    stride = tuple(map(int, stride.split(',')))
    model = TubeletTemporalSpatialPart_concat_chan_2_Transformer(dataset=dataset, embed_dim_ratio=embed_dim, num_frames=segment_length, num_classes=num_classes, num_joints=num_joints, in_chans=in_chans, kernel=kernel, stride=stride, mlp_ratio=2., qkv_bias=True, qk_scale=None, dropout=0.1, pad_mode = 'constant')
elif model_type == 'spatial-temporal':
    model = SpatialTemporalTransformer(embed_dim_ratio=embed_dim, num_frames=segment_length, num_classes=num_classes, num_joints=num_joints, in_chans=in_chans, mlp_ratio=2., qkv_bias=True, qk_scale=None, dropout=0.1)



#Load model state dict
model.load_state_dict(torch.load(NEW_PATH), strict=False)
model.to(device)
model.eval()

# %%
# if cfg['DECOMPOSED']['ENABLE']:
#     if cfg['DECOMPOSED']['TYPE'] == "GR":
#         dec_GR_path = "decom_GR_"
#     elif cfg['DECOMPOSED']['TYPE'] == "GS":
#         dec_GR_path = "decom_"

if dataset=="HRC":
    decomposed = 'something' if dec else ""
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
    decomposed = 'something' if dec else ""

    PIK_train = "/home/s2435462/HRC/data/"+dataset+"/trajectories_train_NTU_"+decomposed+dimension+".dat"
    PIK_test = "/home/s2435462/HRC/data/"+dataset+"/trajectories_test_NTU_"+decomposed+dimension+".dat"
    all_categories = get_NTU_categories()
else:
    raise Exception('dataset not recognized, must be HRC or NTU')

print(PIK_train)
print(PIK_test)

with open(PIK_test, "rb") as f:
    test_crime_trajectories = pickle.load(f)

test_crime_trajectories = remove_short_trajectories(test_crime_trajectories, input_length=segment_length, input_gap=0, pred_length=0)


# %%
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


# test_sample_trajectory= {}

# test_sample_trajectory[test_sample] = test_crime_trajectories[test_sample]

# print(test_sample_trajectory)

# print('\nTest sample %s has %d frames' % (test_sample, len(test_sample_trajectory[test_sample].frames)))

test = TrajectoryDataset(*extract_fixed_sized_segments(dataset, test_crime_trajectories, input_length=segment_length))

number_of_segments = len(test)
print('number_of_segments', number_of_segments)

test_dataloader = torch.utils.data.DataLoader(test, batch_size = 200, shuffle=False, collate_fn=collator_for_lists)

# traj_ids_test, traj_videos_test, traj_persons_test, traj_frames_test, traj_categories_test, X_test = extract_fixed_sized_segments(dataset, test_sample_trajectory, input_length=segment_length)

#Evaluate test sample
# test_dataloader = torch.utils.data.DataLoader([ [traj_categories_test[i], traj_videos_test[i], traj_persons_test[i], traj_frames_test[i], X_test[i] ] for i in range(len(traj_ids_test))], shuffle=False, batch_size=number_of_segments) 
# labels, videos, persons, frames, categories, data = next(iter(test_dataloader)) #test_dataloader consists of only 1 batch



# %%
embeddings = torch.tensor([]).to(device)
classes = torch.LongTensor([]).to(device)

# %%
for iter, batch in enumerate(test_dataloader):
    data, categories = batch['coordinates'], batch['categories']

    labels = torch.tensor([y[0] for y in categories]).to(device)
    # videos = videos.to(device)
    # persons = persons.to(device)
    # frames = frames.to(device)
    data = data.to(device)

    # print(type(data))
    # print(data.shape)
    # break

    outputs = model(data)

    if not os.path.isfile('/home/s2435462/HRC/results/tsne_silhouette' +'/'+filename+'/'+'tsne.pkl'):
        tsne_data = {'embeddings' : outputs.to('cpu'),
                     'classes' : labels.to('cpu')}
        with open('/home/s2435462/HRC/results/tsne_silhouette' +'/'+filename+'/'+'tsne.pkl', 'wb') as fo:
            pickle.dump(tsne_data, fo)
    else:
        with open('/home/s2435462/HRC/results/tsne_silhouette' +'/'+filename+'/'+'tsne.pkl', 'rb') as fo:
            tsne_data = pickle.load(fo)
        with open('/home/s2435462/HRC/results/tsne_silhouette' +'/'+filename+'/'+'tsne.pkl', 'wb') as fo:
            tsne_data['embeddings'] = torch.cat((tsne_data['embeddings'], outputs.to('cpu')), 0)
            tsne_data['classes'] = torch.cat((tsne_data['classes'], labels.to('cpu')), 0)
            pickle.dump(tsne_data, fo)
    
    # embeddings = torch.cat((embeddings, outputs), 0)
    # classes = torch.cat((classes, labels), 0)
    # print(iter)
    if iter% 100 == 0:
        print('iter ', iter)
# %%
# print(embeddings.shape)
# print(classes.shape)

# tsne_data = {'embeddings' : embeddings,
#              'classes' : classes}

# with open('/home/s2435462/HRC/results/tsne_silhouette' +'/'+filename+'/'+'tsne.pkl') as fo:
#     pickle.dump(tsne_data, fo)


