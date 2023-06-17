 #!/bin/env python

import torch
import torch.nn as nn
from einops import rearrange, repeat
import torch.nn.functional as F
import time

from vit import VisionTransformer
from torchvision.models import vit_b_16
from torchvision.models import ViT_B_16_Weights
from torchvision.models import vit_b_32
from torchvision.models import ViT_B_32_Weights
from torchvision.models import resnet18
from torchvision.models import ResNet18_Weights

from torch.nn import MultiheadAttention

def get_average_body_parts(num_joints, x):
    '''
    Returns the midpoint of different body parts
    '''
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
    '''
    Given a keypoint position, returns it from the list of keypoints
    '''
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
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.,return_attention=True):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        
        # NOTE scale factor can be manually set to be compat with prev weights
        self.scale = qk_scale or head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.attn_scores = None

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4) #QKV, B, NUM_HEAD, 21,4
        q, k, v = qkv[0], qkv[1], qkv[2]   # make torchscript happy (cannot use tensor as tuple)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        # print(attn.shape)
        attn = self.attn_drop(attn)
        self.attn_scores = attn
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        # print(x.shape)
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
        
class TemporalTransformer(nn.Module):
    def __init__(self, num_classes=13, num_frames=12, num_joints=17, in_chans=2, embed_dim=64, depth=2,
                 num_heads=4, mlp_ratio=2., qkv_bias=True, qk_scale=None,
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
        print('embed_dim', embed_dim)
        print('in_chans', in_chans)
        print('num_joints', num_joints)
        print('num_heads',num_heads)
        print('depth',depth)
        ### patch embedding
        self.embedding = nn.Linear(num_joints*in_chans, embed_dim)

        ### Additional class token
        self.cls_token = nn.Parameter(torch.zeros(1, embed_dim))

        ### positional embedding including class token
        self.pos_embed = nn.Parameter(torch.zeros(num_frames+1, embed_dim))

        self.pos_drop = nn.Dropout(p=drop_rate)

        #dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule

        self.blocks = nn.ModuleList([
            Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, dropout=dropout
                 #drop_path=dpr[i] #first try a simple dropout instead of drop path
                )
            for i in range(depth)])
        

        self.norm = nn.LayerNorm(embed_dim, eps=1e-6)

         # Representation layer
        '''if representation_size and not distilled:
            self.num_features = representation_size
            self.pre_logits = nn.Sequential(OrderedDict([
                ('fc', nn.Linear(embed_dim, representation_size)),
                ('act', nn.Tanh())
            ]))
        else:
            self.pre_logits = nn.Identity()'''
        #self.pre_logits = nn.Identity()

        # Classifier head(s)
        "Define standard linear + softmax generation step."
        "use learned linear transformation and softmax function to convert the output to predicted class probabilities"
        self.head = nn.Linear(embed_dim, num_classes) #no softmax is used
        
        # initialize weights
        self.init_weights()

        # taken from https://pytorch.org/tutorials/beginner/transformer_tutorial.html
    def init_weights(self):
          initrange = 0.1
          self.embedding.weight.data.uniform_(-initrange, initrange)
          self.head.bias.data.zero_()
          self.head.weight.data.uniform_(-initrange, initrange)

    def forward_features(self, x):

        #print("call forward features")
        # print(f"x shape: {x.shape}")

        x = self.embedding(x)

        #print(f"self cls_token shape: {self.cls_token.shape}")

        #print("expand cls_token")
        cls_token = self.cls_token.expand(x.shape[0], -1, -1)  # stole cls_tokens impl from Phil Wang, thanks

        #print(f"expanded cls_token shape: {cls_token.shape}")
        #print(f"embedded x shape: {x.shape}")

        x = torch.cat((cls_token, x), dim=1)

        #print(f"embedded x + cls_token shape: {x.shape}")

        #print(f"pos_embed shape: {self.pos_embed.shape}")
        #print(f"x + self.pos_embed shape: {(x + self.pos_embed).shape}")

        x = self.pos_drop(x + self.pos_embed)
        # print(f"pos_drop x shape: {x.shape}")

        #x = self.blocks(x)
        for blk in self.blocks:
            x = blk(x)
            #print(f"blocks(x) shape: {x.shape}")

        x = self.norm(x)
        # print("x shape:", x.shape)
        cls_token_final = x[:, 0]
        # print(cls_token_final.size())
        #return self.pre_logits(x[:, 0])
        return cls_token_final
    
    def forward(self, x):
        x = self.forward_features(x)
        # print(x.size())
        x = self.head(x)
        # print(x.size())
        x = F.log_softmax(x, dim=1)
        return x


#input 34 values (one per 2D joint) and the vector describes the window of how the joint value fluctuates 
class TemporalTransformer_2(nn.Module):
    def __init__(self, num_classes=13, num_frames=12, num_joints=17, in_chans=2, embed_dim=64, depth=3,
                 num_heads=4, mlp_ratio=2., qkv_bias=True, qk_scale=None,
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
        print('embed_dim', embed_dim)
        print('in_chans', in_chans)
        print('num_joints', num_joints)

        ### patch embedding
        self.embedding = nn.Linear(num_frames, embed_dim)

        ### Additional class token
        self.cls_token = nn.Parameter(torch.zeros(1, embed_dim))

        ### positional embedding including class token
        self.pos_embed = nn.Parameter(torch.zeros(num_joints*in_chans + 1, embed_dim))

        self.pos_drop = nn.Dropout(p=drop_rate)

        #dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule

        self.blocks = nn.ModuleList([
            Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, dropout=dropout
                 #drop_path=dpr[i] #first try a simple dropout instead of drop path
                )
            for i in range(depth)])
        

        self.norm = nn.LayerNorm(embed_dim, eps=1e-6)

         # Representation layer
        '''if representation_size and not distilled:
            self.num_features = representation_size
            self.pre_logits = nn.Sequential(OrderedDict([
                ('fc', nn.Linear(embed_dim, representation_size)),
                ('act', nn.Tanh())
            ]))
        else:
            self.pre_logits = nn.Identity()'''
        #self.pre_logits = nn.Identity()

        # Classifier head(s)
        "Define standard linear + softmax generation step."
        "use learned linear transformation and softmax function to convert the output to predicted class probabilities"
        self.head = nn.Linear(embed_dim, num_classes) #no softmax is used
        
        # initialize weights
        self.init_weights()

        # taken from https://pytorch.org/tutorials/beginner/transformer_tutorial.html
    def init_weights(self):
          initrange = 0.1
          self.embedding.weight.data.uniform_(-initrange, initrange)
          self.head.bias.data.zero_()
          self.head.weight.data.uniform_(-initrange, initrange)

    def forward_features(self, x):

        #print("call forward features")
        #print(f"x shape: {x.shape}")

        #print('x[0]', x[0])

        x = torch.transpose(x, 1, 2)
        #print('new x.shape', x.shape)

        #print('x[0]', x[0])

        x = self.embedding(x)

        #print(f"self cls_token shape: {self.cls_token.shape}")

        #print("expand cls_token")
        cls_token = self.cls_token.expand(x.shape[0], -1, -1)  # stole cls_tokens impl from Phil Wang, thanks

        #print(f"expanded cls_token shape: {cls_token.shape}")
        #print(f"embedded x shape: {x.shape}")

        x = torch.cat((cls_token, x), dim=1)

        #print(f"embedded x + cls_token shape: {x.shape}")

        #print(f"pos_embed shape: {self.pos_embed.shape}")
        #print(f"x + self.pos_embed shape: {(x + self.pos_embed).shape}")

        x = self.pos_drop(x + self.pos_embed)
        #print(f"pos_drop x shape: {x.shape}")

        for blk in self.blocks:
            x = blk(x)
            #print(f"blocks(x) shape: {x.shape}")

        x = self.norm(x)

        cls_token_final = x[:, 0]
        #return self.pre_logits(x[:, 0])
        return cls_token_final
    
    def forward(self, x):
        x = self.forward_features(x)
        x = self.head(x)
        x = F.log_softmax(x, dim=1)
        return x

#Input frame sequences of average body parts coordinates
class TemporalTransformer_3(nn.Module):
    def __init__(self, num_classes=13, num_frames=12, num_joints=17, num_parts=5, in_chans=2, embed_dim=64, depth=4,
                 num_heads=8, mlp_ratio=2., qkv_bias=True, qk_scale=None,
                 drop_rate=0., attn_drop_rate=0., dropout=0.2):
        """    ##########hybrid_backbone=None, representation_size=None,
        Args:
            num_classes (int): number of classes for classification head, HR-Crime constists of 13 crime categories
            num_frames (int): number of input frames
            num_joints (int): number of joints per skeleton
            num_parts (int): number of body parts in the skeleton
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
        print('embed_dim', embed_dim)
        print('in_chans', in_chans)
        print('num_joints', num_joints)
        print('num_parts', num_parts)

        self.num_joints = num_joints

        ### patch embedding
        self.embedding = nn.Linear(num_parts*in_chans, embed_dim)

        ### Additional class token
        self.cls_token = nn.Parameter(torch.zeros(1, embed_dim))

        ### positional embedding including class token
        self.pos_embed = nn.Parameter(torch.zeros(num_frames+1, embed_dim))

        self.pos_drop = nn.Dropout(p=drop_rate)

        #dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule

        self.blocks = nn.ModuleList([
            Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, dropout=dropout
                 #drop_path=dpr[i] #first try a simple dropout instead of drop path
                )
            for i in range(depth)])
        

        self.norm = nn.LayerNorm(embed_dim, eps=1e-6)

         # Representation layer
        '''if representation_size and not distilled:
            self.num_features = representation_size
            self.pre_logits = nn.Sequential(OrderedDict([
                ('fc', nn.Linear(embed_dim, representation_size)),
                ('act', nn.Tanh())
            ]))
        else:
            self.pre_logits = nn.Identity()'''
        #self.pre_logits = nn.Identity()

        # Classifier head(s)
        "Define standard linear + softmax generation step."
        "use learned linear transformation and softmax function to convert the output to predicted class probabilities"
        self.head = nn.Linear(embed_dim, num_classes) #no softmax is used
        
        # initialize weights
        self.init_weights()

        # taken from https://pytorch.org/tutorials/beginner/transformer_tutorial.html
    def init_weights(self):
          initrange = 0.1
          self.embedding.weight.data.uniform_(-initrange, initrange)
          self.head.bias.data.zero_()
          self.head.weight.data.uniform_(-initrange, initrange)

    def forward_features(self, x):

        #print("call forward features")
        #print(f"x shape: {x.shape}")

        x = self.embedding(x)

        #print(f"self cls_token shape: {self.cls_token.shape}")

        #print("expand cls_token")
        cls_token = self.cls_token.expand(x.shape[0], -1, -1)  # stole cls_tokens impl from Phil Wang, thanks

        #print(f"expanded cls_token shape: {cls_token.shape}")
        #print(f"embedded x shape: {x.shape}")

        x = torch.cat((cls_token, x), dim=1)

        #print(f"embedded x + cls_token shape: {x.shape}")

        #print(f"pos_embed shape: {self.pos_embed.shape}")
        #print(f"x + self.pos_embed shape: {(x + self.pos_embed).shape}")

        # x = self.pos_drop(x + self.pos_embed)
        #print(f"pos_drop x shape: {x.shape}")

        #x = self.blocks(x)
        for blk in self.blocks:
            x = blk(x)
            #print(f"blocks(x) shape: {x.shape}")

        x = self.norm(x)

        cls_token_final = x[:, 0]
        #return self.pre_logits(x[:, 0])
        return cls_token_final
    
    def forward(self, x):
        
        x = get_average_body_parts(self.num_joints, x)
        #print('x[0]', x[0])

        #print('x.shape', x.shape)

        x = self.forward_features(x)
        x = self.head(x)
        x = F.log_softmax(x, dim=1)
        return x


#input 10 (5x2 one per x and one per y per body part)[xAVG_torso yAVG_torso , etc] - mixing the input 17 and the body part information
class TemporalTransformer_4(nn.Module):
    def __init__(self, num_classes=13, num_frames=12, num_joints=17, num_parts=5, in_chans=2, embed_dim=64, depth=4,
                 num_heads=8, mlp_ratio=2., qkv_bias=True, qk_scale=None,
                 drop_rate=0., attn_drop_rate=0., dropout=0.2):
        """    ##########hybrid_backbone=None, representation_size=None,
        Args:
            num_classes (int): number of classes for classification head, HR-Crime constists of 13 crime categories
            num_frames (int): number of input frames
            num_joints (int): number of joints per skeleton
            num_parts (int): number of body parts in the skeleton
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
        print('embed_dim', embed_dim)
        print('in_chans', in_chans)
        print('num_joints', num_joints)
        print('num_parts', num_parts)

        self.num_joints = num_joints
        ### patch embedding
        self.embedding = nn.Linear(num_frames, embed_dim)

        ### Additional class token
        self.cls_token = nn.Parameter(torch.zeros(1, embed_dim))

        ### positional embedding including class token
        self.pos_embed = nn.Parameter(torch.zeros(num_parts*in_chans + 1, embed_dim))

        self.pos_drop = nn.Dropout(p=drop_rate)

        #dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule

        self.blocks = nn.ModuleList([
            Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, dropout=dropout
                 #drop_path=dpr[i] #first try a simple dropout instead of drop path
                )
            for i in range(depth)])
        

        self.norm = nn.LayerNorm(embed_dim, eps=1e-6)

         # Representation layer
        '''if representation_size and not distilled:
            self.num_features = representation_size
            self.pre_logits = nn.Sequential(OrderedDict([
                ('fc', nn.Linear(embed_dim, representation_size)),
                ('act', nn.Tanh())
            ]))
        else:
            self.pre_logits = nn.Identity()'''
        #self.pre_logits = nn.Identity()

        # Classifier head(s)
        "Define standard linear + softmax generation step."
        "use learned linear transformation and softmax function to convert the output to predicted class probabilities"
        self.head = nn.Linear(embed_dim, num_classes) #no softmax is used
        
        # initialize weights
        self.init_weights()

        # taken from https://pytorch.org/tutorials/beginner/transformer_tutorial.html
    def init_weights(self):
          initrange = 0.1
          self.embedding.weight.data.uniform_(-initrange, initrange)
          self.head.bias.data.zero_()
          self.head.weight.data.uniform_(-initrange, initrange)

    def forward_features(self, x):

        #print("call forward features")
        #print(f"x shape: {x.shape}")

        #print('x[0]', x[0])

        x = torch.transpose(x, 1, 2)
        #print('new x.shape', x.shape)

        #print('x[0]', x[0])

        x = self.embedding(x)

        #print(f"self cls_token shape: {self.cls_token.shape}")

        #print("expand cls_token")
        cls_token = self.cls_token.expand(x.shape[0], -1, -1)  # stole cls_tokens impl from Phil Wang, thanks

        #print(f"expanded cls_token shape: {cls_token.shape}")
        #print(f"embedded x shape: {x.shape}")

        x = torch.cat((cls_token, x), dim=1)

        #print(f"embedded x + cls_token shape: {x.shape}")

        #print(f"pos_embed shape: {self.pos_embed.shape}")
        #print(f"x + self.pos_embed shape: {(x + self.pos_embed).shape}")

        x = self.pos_drop(x + self.pos_embed)
        print(f"pos_drop x shape: {x.shape}")
        # print(x.size())
        for blk in self.blocks:
            x = blk(x)
            #print(f"blocks(x) shape: {x.shape}")

        x = self.norm(x)

        cls_token_final = x[:, 0]
        #return self.pre_logits(x[:, 0])

        return cls_token_final
    
    def forward(self, x):
        
        x = get_average_body_parts(self.num_joints, x)
        #print('x[0]', x[0])

        #print('x.shape', x.shape)

        x = self.forward_features(x)
        x = self.head(x)
        x = F.log_softmax(x, dim=1)
        return x
        

class SpatialTemporalTransformer(nn.Module):
    '''
    ST-Tran Transformer
    '''
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
        
        x = self.head(x)
        x = F.log_softmax(x, dim=1)


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




class BodyPartTransformer(nn.Module):
    '''
    SBPT-Tran Transformer
    '''
    def __init__(self, dataset=None, num_classes=13, num_frames=12, num_joints=17, in_chans=2, embed_dim_ratio=32, depth=4,
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

        self.in_chans = in_chans
        self.dataset = dataset

        embed_dim = embed_dim_ratio * num_joints   #### temporal embed_dim is num_joints * spatial embedding dim ratio

        #self.embedding = nn.Linear(num_joints*2, embed_dim)
        #self.pos_embed = nn.Parameter(torch.zeros(num_frames+1, embed_dim))

        ### spatial patch embedding
        self.Spatial_patch_to_embedding = nn.Linear(in_chans, embed_dim_ratio)
        #self.Spatial_pos_embed = nn.Parameter(torch.zeros(num_joints, embed_dim_ratio))
        if self.dataset == "HRC":
            self.Torso_pos_embed = nn.Parameter(torch.zeros(9, embed_dim_ratio)) #9 joints
            self.Other_pos_embed = nn.Parameter(torch.zeros(2, embed_dim_ratio)) #2 joints in remaining body parts
        elif "NTU" in self.dataset:
            if "2D" in self.dataset:
                self.Torso_pos_embed = nn.Parameter(torch.zeros(9, embed_dim_ratio)) #9 joints
                self.Elbow_Knee_pos_embed = nn.Parameter(torch.zeros(2, embed_dim_ratio)) #2 joints in remaining body parts 
                self.Wrist_pos_embed = nn.Parameter(torch.zeros(8, embed_dim_ratio)) #2 joints in remaining body parts 
                self.Ankle_pos_embed = nn.Parameter(torch.zeros(4, embed_dim_ratio)) #2 joints in remaining body parts 


        self.Temporal_pos_embed = nn.Parameter(torch.zeros(num_frames + 1, embed_dim))
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
          self.Spatial_patch_to_embedding.weight.data.uniform_(-initrange, initrange)
          self.head.bias.data.zero_()
          self.head.weight.data.uniform_(-initrange, initrange)
    
    def Torso_forward_features(self, x):
        #print('\nCall Torso_forward_features')
        #print('x.shape', x.shape)
        b, f, p, c = x.shape  ##### b is batch size, f is number of frames, p is number of joints
        x = rearrange(x, 'b f p c -> (b f) p c', ) ####concatenate coordinates along frames and reorder axes

        #print('new x.shape', x.shape)

        x = self.Spatial_patch_to_embedding(x)
        x = self.pos_drop(x + self.Torso_pos_embed)

        for blk in self.Torso_blocks:
            x = blk(x)

        x = self.Spatial_norm(x)
        x = rearrange(x, '(b f) w c -> b f (w c)', f=f) ####rearrange tensor to match temporal transformer input shape [batch_size, num_frames, embed_dim]

        #print('rearranged x.shape', x.shape)
        return x
    
    def Elbow_forward_features(self, x):
        #print('\nCall Elbow_forward_features')
        #print('x.shape', x.shape)
        b, f, p, c = x.shape  ##### b is batch size, f is number of frames, p is number of joints
        x = rearrange(x, 'b f p c -> (b f) p c', ) ####concatenate coordinates along frames and reorder axes

        #print('new x.shape', x.shape)

        x = self.Spatial_patch_to_embedding(x)

        if "NTU" in self.dataset:
            x = self.pos_drop(x + self.Elbow_Knee_pos_embed)
        else:
            x = self.pos_drop(x + self.Other_pos_embed)

        
        for blk in self.Elbow_blocks:
            x = blk(x)

        x = self.Spatial_norm(x)
        x = rearrange(x, '(b f) w c -> b f (w c)', f=f) ####rearrange tensor to match temporal transformer input shape [batch_size, num_frames, embed_dim]

        #print('rearranged x.shape', x.shape)
        return x
    
    def Wrist_forward_features(self, x):
        #print('\nCall Wrist_forward_features')
        #print('x.shape', x.shape)
        b, f, p, c = x.shape  ##### b is batch size, f is number of frames, p is number of joints
        x = rearrange(x, 'b f p c -> (b f) p c', ) ####concatenate coordinates along frames and reorder axes

        #print('new x.shape', x.shape)

        x = self.Spatial_patch_to_embedding(x)

        if "NTU" in self.dataset:
            x = self.pos_drop(x + self.Wrist_pos_embed)
        else:
            x = self.pos_drop(x + self.Other_pos_embed)

        for blk in self.Wrist_blocks:
            x = blk(x)

        x = self.Spatial_norm(x)
        x = rearrange(x, '(b f) w c -> b f (w c)', f=f) ####rearrange tensor to match temporal transformer input shape [batch_size, num_frames, embed_dim]

        #print('rearranged x.shape', x.shape)
        return x
    
    def Knee_forward_features(self, x):
        #print('\nCall Knee_forward_features')
        #print('x.shape', x.shape)
        b, f, p, c = x.shape  ##### b is batch size, f is number of frames, p is number of joints
        x = rearrange(x, 'b f p c -> (b f) p c', ) ####concatenate coordinates along frames and reorder axes

        #print('new x.shape', x.shape)

        x = self.Spatial_patch_to_embedding(x)
        if "NTU" in self.dataset:
            x = self.pos_drop(x + self.Elbow_Knee_pos_embed)
        else:
            x = self.pos_drop(x + self.Other_pos_embed)

        for blk in self.Knee_blocks:
            x = blk(x)

        x = self.Spatial_norm(x)
        x = rearrange(x, '(b f) w c -> b f (w c)', f=f) ####rearrange tensor to match temporal transformer input shape [batch_size, num_frames, embed_dim]

        #print('rearranged x.shape', x.shape)
        return x

    def Ankle_forward_features(self, x):
        #print('\nCall Ankle_forward_features')
        #print('x.shape', x.shape)
        b, f, p, c = x.shape  ##### b is batch size, f is number of frames, p is number of joints
        x = rearrange(x, 'b f p c -> (b f) p c', ) ####concatenate coordinates along frames and reorder axes

        #print('new x.shape', x.shape)

        x = self.Spatial_patch_to_embedding(x)
        
        if "NTU" in self.dataset:
            x = self.pos_drop(x + self.Ankle_pos_embed)
        else:
            x = self.pos_drop(x + self.Other_pos_embed)

        for blk in self.Ankle_blocks:
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
        b, f, e = x.shape  ##### b is batch size, f is number of frames, e is number of elements equal to 2xnumber of joints
        c = self.in_chans ##### number of channels, in our case 2
        #print('b %d, f %d, e %d' %(b,f,e))
        #print('c',c)
        j = e//c ##### number of joints
        #print('j',j)
        x = torch.reshape(x, (b, f, j, c))
        #print('x.shape', x.shape)
       
        ### now x is [batch_size, 2 channels, receptive frames, joint_num], following image data
        #print('x.shape following image data', x.shape)
        #print('x[0,:]', x[0,:])
        if "HRC" in self.dataset:
            x_torso_1 = x[:, :, 0:7, :] #joints 0,1,2,3,4,5,6 (head and shoulders) 
            x_torso_2 = x[:, :, 11:13, :] #joints 11,12 (hips)
            x_torso = torch.cat((x_torso_1, x_torso_2), dim=2)
            x_elbow = x[:, :, 7:9, :]
            x_wrist = x[:, :, 9:11, :]
            x_knee = x[:, :, 13:15, :]
            x_ankle = x[:, :, 15:17, :]
        elif "NTU" in self.dataset:
            if "2D" in self.dataset:
                x_torso_1 = x[:, :, 0:5, :]
                x_torso_2 = x[:, :, 8:9, :]
                x_torso_3 = x[:, :, 12:13, :]
                x_torso_4 = x[:, :, 16:17, :]
                x_torso_5 = x[:, :, 20:21, :]

                x_torso = torch.cat((x_torso_1, x_torso_2, x_torso_3, x_torso_4, x_torso_5), dim=2)

                x_elbow_1 = x[:, :, 9:10, :]
                x_elbow_2 = x[:, :, 5:6, :]

                x_elbow = torch.cat((x_elbow_1, x_elbow_2), dim=2)

                x_wrist_1 = x[:, :, 7:9, :]
                x_wrist_2 = x[:, :, 10:12, :]
                x_wrist_3 = x[:, :, 21:25, :]

                x_wrist = torch.cat((x_wrist_1, x_wrist_2, x_wrist_3), dim=2)

                x_knee_1 = x[:, :, 15:16, :]
                x_knee_2 = x[:, :, 17:18, :]

                x_knee = torch.cat((x_knee_1, x_knee_2), dim=2)

                x_ankle_1 = x[:, :, 14:16, :]
                x_ankle_2 = x[:, :, 18:20, :]

                x_ankle = torch.cat((x_ankle_1, x_ankle_2), dim=2)


        
        '''
        print('x_torso shape', x_torso.shape)
        print('x_elbow shape', x_elbow.shape)
        print('x_wirst shape', x_wirst.shape)
        print('x_knee shape', x_knee.shape)
        print('x_ankle shape', x_ankle.shape)
        
        
        print('x_torso', x_torso)
        print('x_elbow', x_elbow)
        print('x_wirst', x_wirst)
        print('x_knee', x_knee)
        print('x_ankle', x_ankle)
        '''
        
        x_torso = self.Torso_forward_features(x_torso)
        x_elbow = self.Elbow_forward_features(x_elbow)
        x_wrist = self.Wrist_forward_features(x_wrist)
        x_knee = self.Knee_forward_features(x_knee)
        x_ankle = self.Ankle_forward_features(x_ankle)

        '''
        print('x_torso features shape', x_torso.shape)
        print('x_elbow features shape', x_elbow.shape)
        print('x_wirst features shape', x_wirst.shape)
        print('x_knee features shape', x_knee.shape)
        print('x_ankle features shape', x_ankle.shape)
        '''

        #print('x_torso[0]', x_torso[0])
        #print('x_elbow[0]', x_elbow[0])

        x = torch.cat((x_torso, x_elbow, x_wrist, x_knee, x_ankle), dim=2)
        #print('x[0]', x[0])

        #print('x.shape', x.shape)

        x = self.forward_features(x)
        
        x = self.head(x)

        x = F.log_softmax(x, dim=1)

        #print(f"head(x) size: {x.size()}")

        return x
#End of kaylieh################


#Start Ajay ############################################################### 

class TubeletTemporalTransformer(nn.Module):
    '''
    TTubeFormer Transformer
    '''
    def __init__(self, dataset=None, num_classes=13, num_frames=12, num_joints=17, in_chans=2, embed_dim=64, kernel=None, stride=None, depth=4,
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
        print('embed_dim', embed_dim)
        print('in_chans', in_chans)
        print('num_joints', num_joints)
        
        self.dataset = dataset
        self.c3d = torch.nn.Conv3d(in_chans, embed_dim, kernel_size=kernel, stride=stride)

        ### patch embedding
        # self.embedding = nn.Linear(num_joints*in_chans, embed_dim)

        ### Additional class token
        self.cls_token = nn.Parameter(torch.zeros(1, embed_dim))

        ### positional embedding including class token
        if "NTU" in self.dataset:
            num_pos = int(num_frames/5) * int(5/2) * int(5/2)
        elif "HRC" in self.dataset:
            num_pos = int(num_frames/5) * int(5/2) * int(5/2)
        
        self.pos_embed = nn.Parameter(torch.zeros(num_pos+1, embed_dim))

        self.pos_drop = nn.Dropout(p=drop_rate)

        self.blocks = nn.ModuleList([
            Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, dropout=dropout
                )
            for i in range(depth)])
        

        self.norm = nn.LayerNorm(embed_dim, eps=1e-6)

         # Representation layer
        '''if representation_size and not distilled:
            self.num_features = representation_size
            self.pre_logits = nn.Sequential(OrderedDict([
                ('fc', nn.Linear(embed_dim, representation_size)),
                ('act', nn.Tanh())
            ]))
        else:
            self.pre_logits = nn.Identity()'''
        #self.pre_logits = nn.Identity()

        # Classifier head(s)
        "Define standard linear + softmax generation step."
        "use learned linear transformation and softmax function to convert the output to predicted class probabilities"
        self.head = nn.Linear(embed_dim, num_classes) #no softmax is used
        
        # initialize weights
        self.init_weights()

        # taken from https://pytorch.org/tutorials/beginner/transformer_tutorial.html
    def init_weights(self):
          initrange = 0.1
        #   self.embedding.weight.data.uniform_(-initrange, initrange)
          self.head.bias.data.zero_()
          self.head.weight.data.uniform_(-initrange, initrange)

    def forward_features(self, x):
        
        if "NTU" in self.dataset:
            x = rearrange(x, 'b f (h w c) -> b c f h w', h=5, w=5, c=2)
        elif "HRC" in self.dataset:
            x = F.pad(x, (0,16))
            x = rearrange(x, 'b f (h w c) -> b c f h w', h=5, w=5, c=2)

        x = self.c3d(x)

        x = rearrange(x, 'b o_c d h w -> b (d h w) o_c')
        # x = self.embedding(x)

        cls_token = self.cls_token.expand(x.shape[0], -1, -1)  # stole cls_tokens impl from Phil Wang, thanks

        x = torch.cat((cls_token, x), dim=1)

        x = self.pos_drop(x + self.pos_embed)

        for blk in self.blocks:
            x = blk(x)

        x = self.norm(x)
        cls_token_final = x[:, 0]
        return cls_token_final
    
    def forward(self, x):
        x = self.forward_features(x)
        x = self.head(x)
        x = F.log_softmax(x, dim=1)
        return x

class TubeletTemporalPart_mean_chan_1_Transformer(nn.Module):
    '''
    BPFormer-V1
    '''
    def __init__(self, dataset=None, num_classes=13, num_frames=12, num_joints=17, in_chans=2, embed_dim=64, kernel=None, stride=None, depth=4,
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
        print('embed_dim', embed_dim)
        print('in_chans', in_chans)
        print('num_joints', num_joints)

        self.dataset = dataset
        
        if "NTU" in dataset or "HRC" in dataset:
            self.torso_conv = torch.nn.Conv3d(1, embed_dim, kernel_size=kernel, stride=stride)
            self.elbows_conv = torch.nn.Conv3d(1, embed_dim, kernel_size=kernel, stride=stride)
            self.wrists_conv = torch.nn.Conv3d(1, embed_dim, kernel_size=kernel, stride=stride)
            self.knees_conv = torch.nn.Conv3d(1, embed_dim, kernel_size=kernel, stride=stride)
            self.ankles_conv = torch.nn.Conv3d(1, embed_dim, kernel_size=kernel, stride=stride)

        ### patch embedding
        # self.embedding = nn.Linear(num_joints*in_chans, embed_dim)

        ### Additional class token
        self.cls_token = nn.Parameter(torch.zeros(1, embed_dim))

        ### positional embedding including class token
        # num_pos = int(num_frames/5) * int(5/2) * int(5/2)
        
        self.pos_embed = nn.Parameter(torch.zeros(5+1, embed_dim)) ## 5 body parts + 1 pos embed

        self.pos_drop = nn.Dropout(p=drop_rate)

        self.blocks = nn.ModuleList([
            Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, dropout=dropout
                )
            for i in range(depth)])
        

        self.norm = nn.LayerNorm(embed_dim, eps=1e-6)

         # Representation layer
        '''if representation_size and not distilled:
            self.num_features = representation_size
            self.pre_logits = nn.Sequential(OrderedDict([
                ('fc', nn.Linear(embed_dim, representation_size)),
                ('act', nn.Tanh())
            ]))
        else:
            self.pre_logits = nn.Identity()'''
        #self.pre_logits = nn.Identity()

        # Classifier head(s)
        "Define standard linear + softmax generation step."
        "use learned linear transformation and softmax function to convert the output to predicted class probabilities"
        self.head = nn.Linear(embed_dim, num_classes) #no softmax is used
        
        # initialize weights
        self.init_weights()

        # taken from https://pytorch.org/tutorials/beginner/transformer_tutorial.html
    def init_weights(self):
          initrange = 0.1
        #   self.embedding.weight.data.uniform_(-initrange, initrange)
          self.head.bias.data.zero_()
          self.head.weight.data.uniform_(-initrange, initrange)

    def tubelet_embedding(self, x):
        if self.dataset == "NTU_3D":
            torso = get_keypoint(x, [], 3)
        elif self.dataset == "NTU_2D":
            torso = get_keypoint(x, [4, 3, 9, 21, 5, 2, 17, 1, 13], 2)
            elbows = get_keypoint(x, [10, 6], 2)
            wrists = get_keypoint(x, [11, 12, 24, 25, 7, 8, 22, 23], 2)
            knees = get_keypoint(x, [18, 14], 2)
            ankles = get_keypoint(x, [19, 20, 15, 16], 2)

            torso = rearrange(torso, "b f (x y) -> b f x y", x= 3, y =6)       ## shape: b f 3 6
            torso = F.pad(input=torso, pad=(0, 0, 2, 1), mode='constant', value=0)   ## Shape: b f 6 6 
            torso = torso.unsqueeze(1)                                         ## Shape: b 1 f 6 6

            elbows = rearrange(elbows, "b f (x y) -> b f x y", x= 2, y =2)     ## Shape: b f 2 2 
            # elbows = F.pad(input=elbows, pad=(1, 1), mode='constant', value=0) ## Already a square
            elbows = elbows.unsqueeze(1)

            wrists = rearrange(wrists, "b f (x y) -> b f x y", x= 4, y =4)     ## Shape: b f 4 4
            # wrists = F.pad(input=wrists, pad=(1, 0), mode='constant', value=0) ## Shape: b f 4 4
            wrists = wrists.unsqueeze(1)                                       ## Shape: b 1 f 4 4

            knees = rearrange(knees, "b f (x y) -> b f x y", x= 2, y =2)       ## Shape: b f 2 2
            # knees = F.pad(input=knees, pad=(2, 1), mode='constant', value=0) ## Already a square
            knees = knees.unsqueeze(1)                                         ## Shape: b 1 f 2 2

            ankles = rearrange(ankles, "b f (x y) -> b f x y", x= 4, y =2)     ## Shape: b f 4 2
            ankles = F.pad(input=ankles, pad=(1, 1), mode='constant', value=0) ## Shape: b f 4 4
            ankles = ankles.unsqueeze(1)                                       ## Shape: b 1 f 4 4

            torso_embed = self.torso_conv(torso)
            elbows_embed = self.elbows_conv(elbows)
            wrists_embed = self.wrists_conv(wrists)
            knees_embed = self.knees_conv(knees)
            ankles_embed = self.ankles_conv(ankles)

            # print(torso_embed.shape)
            # print(elbows_embed.shape)

            torso_embed = torch.flatten(torso_embed, start_dim=2)
            elbows_embed = torch.flatten(elbows_embed, start_dim=2)
            wrists_embed = torch.flatten(wrists_embed, start_dim=2)
            knees_embed = torch.flatten(knees_embed, start_dim=2)
            ankles_embed = torch.flatten(ankles_embed, start_dim=2)


            torso_embed = torch.mean(torso_embed, dim=2)
            elbows_embed = torch.mean(elbows_embed, dim=2)
            wrists_embed = torch.mean(wrists_embed, dim=2)
            knees_embed = torch.mean(knees_embed, dim=2)
            ankles_embed = torch.mean(ankles_embed, dim=2)

            # print(torso_embed.shape)
            # print(elbows_embed.shape)

            x = torch.stack((torso_embed, elbows_embed, wrists_embed, knees_embed, ankles_embed), dim=1)

            return x
        elif self.dataset == "HRC":
            torso = get_keypoint(x, [1, 2, 3, 4, 5, 6, 7, 12, 13], 2)
            elbows = get_keypoint(x, [8, 9], 2)
            wrists = get_keypoint(x, [10, 11], 2)
            knees = get_keypoint(x, [14, 15], 2)
            ankles = get_keypoint(x, [16, 17], 2)

            torso = rearrange(torso, "b f (x y) -> b f x y", x= 3, y =6)       ## shape: b f 6 3
            torso = F.pad(input=torso, pad=(0, 0, 2, 1), mode='constant', value=0)   ## Shape: b f 6 6 
            torso = torso.unsqueeze(1)                                         ## Shape: b 1 f 6 6

            elbows = rearrange(elbows, "b f (x y) -> b f x y", x= 2, y =2)     ## Shape: b f 2 2 
            # elbows = F.pad(input=elbows, pad=(1, 1), mode='constant', value=0) ## Already a square
            elbows = elbows.unsqueeze(1)

            wrists = rearrange(wrists, "b f (x y) -> b f x y", x= 2, y =2)     ## Shape: b f 2 2
            # wrists = F.pad(input=wrists, pad=(1, 0), mode='constant', value=0) ## Already a square
            wrists = wrists.unsqueeze(1)                                       ## Shape: b 1 f 5 5

            knees = rearrange(knees, "b f (x y) -> b f x y", x= 2, y =2)       ## Shape: b f 2 2
            # knees = F.pad(input=knees, pad=(2, 1), mode='constant', value=0) ## Already a square
            knees = knees.unsqueeze(1)                                         ## Shape: b 1 f 2 2

            ankles = rearrange(ankles, "b f (x y) -> b f x y", x= 2, y =2)     ## Shape: b f 4 2
            # ankles = F.pad(input=ankles, pad=(1, 1), mode='constant', value=0) ## Already a square
            ankles = ankles.unsqueeze(1)                                       ## Shape: b 1 f 4 4

            torso_embed = self.torso_conv(torso)
            elbows_embed = self.elbows_conv(elbows)
            wrists_embed = self.wrists_conv(wrists)
            knees_embed = self.knees_conv(knees)
            ankles_embed = self.ankles_conv(ankles)

            # print(torso_embed.shape)
            # print(elbows_embed.shape)

            torso_embed = torch.flatten(torso_embed, start_dim=2)
            elbows_embed = torch.flatten(elbows_embed, start_dim=2)
            wrists_embed = torch.flatten(wrists_embed, start_dim=2)
            knees_embed = torch.flatten(knees_embed, start_dim=2)
            ankles_embed = torch.flatten(ankles_embed, start_dim=2)


            torso_embed = torch.mean(torso_embed, dim=2)
            elbows_embed = torch.mean(elbows_embed, dim=2)
            wrists_embed = torch.mean(wrists_embed, dim=2)
            knees_embed = torch.mean(knees_embed, dim=2)
            ankles_embed = torch.mean(ankles_embed, dim=2)

            # print(torso_embed.shape)
            # print(elbows_embed.shape)

            x = torch.stack((torso_embed, elbows_embed, wrists_embed, knees_embed, ankles_embed), dim=1)

            return x
            

    def forward_features(self, x):
        
        # x = rearrange(x, 'b f (h w c) -> b c f h w', h=5, w=5, c=2)
        x = self.tubelet_embedding(x)
        # x = self.c3d(x)

        # x = rearrange(x, 'b o_c d h w -> b (d h w) o_c')
        # x = self.embedding(x)

        cls_token = self.cls_token.expand(x.shape[0], -1, -1)  # stole cls_tokens impl from Phil Wang, thanks
        # print(x.shape)
        # print(cls_token.shape)
        x = torch.cat((cls_token, x), dim=1)

        x = self.pos_drop(x + self.pos_embed)

        for blk in self.blocks:
            x = blk(x)

        x = self.norm(x)
        cls_token_final = x[:, 0]
        return cls_token_final
    
    def forward(self, x):
        x = self.forward_features(x)
        x = self.head(x)
        x = F.log_softmax(x, dim=1)
        return x

class TubeletTemporalPart_concat_chan_1_Transformer(nn.Module):
    '''
    BPFormer-V2
    '''
    ## Instead of mean, concatenate with a smaller dimension. 
    ## Also means you have to pad to make same dimension as largest input square. 
    ## Otherwise, the embeddings will have different lengths after concatenating.
    def __init__(self, dataset=None, num_classes=13, num_frames=12, num_joints=17, in_chans=2, embed_dim=64, kernel=None, stride=None, depth=4,
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
        print('embed_dim', embed_dim)
        print('in_chans', in_chans)
        print('num_joints', num_joints)

        self.dataset = dataset
        
        if "NTU" in dataset or "HRC" in dataset:
            self.torso_conv = torch.nn.Conv3d(1, embed_dim, kernel_size=kernel, stride=stride)
            self.elbows_conv = torch.nn.Conv3d(1, embed_dim, kernel_size=kernel, stride=stride)
            self.wrists_conv = torch.nn.Conv3d(1, embed_dim, kernel_size=kernel, stride=stride)
            self.knees_conv = torch.nn.Conv3d(1, embed_dim, kernel_size=kernel, stride=stride)
            self.ankles_conv = torch.nn.Conv3d(1, embed_dim, kernel_size=kernel, stride=stride)

        ### patch embedding
        # self.embedding = nn.Linear(num_joints*in_chans, embed_dim)

        num_embed = (int((num_frames-kernel[0])/stride[0]) + 1) * (int((6-kernel[1])/stride[1]) + 1) * (int((6-kernel[2])/stride[2]) + 1)
        
        self.final_embed_dim = num_embed * embed_dim
        ### Additional class token
        self.cls_token = nn.Parameter(torch.zeros(1, self.final_embed_dim))

        ### positional embedding including class token
        # num_pos = int(num_frames/5) * int(5/2) * int(5/2)

        
        
        

        self.pos_embed = nn.Parameter(torch.zeros(5+1, self.final_embed_dim)) ## 5 body parts + 1 pos embed

        self.pos_drop = nn.Dropout(p=drop_rate)

        self.blocks = nn.ModuleList([
            Block(
                dim=self.final_embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, dropout=dropout
                )
            for i in range(depth)])
        

        self.norm = nn.LayerNorm(self.final_embed_dim, eps=1e-6)

         # Representation layer
        '''if representation_size and not distilled:
            self.num_features = representation_size
            self.pre_logits = nn.Sequential(OrderedDict([
                ('fc', nn.Linear(embed_dim, representation_size)),
                ('act', nn.Tanh())
            ]))
        else:
            self.pre_logits = nn.Identity()'''
        #self.pre_logits = nn.Identity()

        # Classifier head(s)
        "Define standard linear + softmax generation step."
        "use learned linear transformation and softmax function to convert the output to predicted class probabilities"
        self.head = nn.Linear(self.final_embed_dim, num_classes) #no softmax is used
        
        # initialize weights
        self.init_weights()

        # taken from https://pytorch.org/tutorials/beginner/transformer_tutorial.html
    def init_weights(self):
          initrange = 0.1
        #   self.embedding.weight.data.uniform_(-initrange, initrange)
          self.head.bias.data.zero_()
          self.head.weight.data.uniform_(-initrange, initrange)

    def tubelet_embedding(self, x):
        if self.dataset == "NTU_3D":
            torso = get_keypoint(x, [], 3)
        elif self.dataset == "NTU_2D":
            torso = get_keypoint(x, [4, 3, 9, 21, 5, 2, 17, 1, 13], 2)
            elbows = get_keypoint(x, [10, 6], 2)
            wrists = get_keypoint(x, [11, 12, 24, 25, 7, 8, 22, 23], 2)
            knees = get_keypoint(x, [18, 14], 2)
            ankles = get_keypoint(x, [19, 20, 15, 16], 2)

            torso = rearrange(torso, "b f (x y) -> b f x y", x= 3, y =6)       ## shape: b f 6 3
            torso = F.pad(input=torso, pad=(0, 0, 2, 1), mode='constant', value=0)   ## Shape: b f 6 6 
            torso = torso.unsqueeze(1)                                         ## Shape: b 1 f 6 6

            elbows = rearrange(elbows, "b f (x y) -> b f x y", x= 2, y =2)     ## Shape: b f 2 2 
            elbows = F.pad(input=elbows, pad=(2, 2, 2, 2), mode='constant', value=0) ## Shape: b f 6 6
            elbows = elbows.unsqueeze(1)

            wrists = rearrange(wrists, "b f (x y) -> b f x y", x= 4, y =4)     ## Shape: b f 4 4
            wrists = F.pad(input=wrists, pad=(1, 1, 1, 1), mode='constant', value=0) ## Shape: b f 6 6
            wrists = wrists.unsqueeze(1)                                       ## Shape: b 1 f 6 6

            knees = rearrange(knees, "b f (x y) -> b f x y", x= 2, y =2)       ## Shape: b f 2 2
            knees = F.pad(input=knees, pad=(2, 2, 2, 2), mode='constant', value=0) ## Shape: b f 6 6
            knees = knees.unsqueeze(1)                                         ## Shape: b 1 f 6 6

            ankles = rearrange(ankles, "b f (x y) -> b f x y", x= 4, y =2)     ## Shape: b f 4 2
            ankles = F.pad(input=ankles, pad=(2, 2, 1, 1), mode='constant', value=0) ## Shape: b f 6 6
            ankles = ankles.unsqueeze(1)                                       ## Shape: b 1 f 6 6

            torso_embed = self.torso_conv(torso)
            elbows_embed = self.elbows_conv(elbows)
            wrists_embed = self.wrists_conv(wrists)
            knees_embed = self.knees_conv(knees)
            ankles_embed = self.ankles_conv(ankles)

            # print(torso_embed.shape)
            # print(elbows_embed.shape)

            torso_embed = torch.flatten(torso_embed, start_dim=2)
            elbows_embed = torch.flatten(elbows_embed, start_dim=2)
            wrists_embed = torch.flatten(wrists_embed, start_dim=2)
            knees_embed = torch.flatten(knees_embed, start_dim=2)
            ankles_embed = torch.flatten(ankles_embed, start_dim=2)

            ## CONCAT THE RESULTS OF CONVOLUTION (N x 32 dimensional)
            torso_embed = rearrange(torso_embed, "b e n -> b (n e)")
            elbows_embed = rearrange(elbows_embed, "b e n -> b (n e)")
            wrists_embed = rearrange(wrists_embed, "b e n -> b (n e)")
            knees_embed = rearrange(knees_embed, "b e n -> b (n e)")
            ankles_embed = rearrange(ankles_embed, "b e n -> b (n e)")

            # print(torso_embed.shape)
            # print(elbows_embed.shape)

            x = torch.stack((torso_embed, elbows_embed, wrists_embed, knees_embed, ankles_embed), dim=1)

            return x
        elif self.dataset == "HRC":
            torso = get_keypoint(x, [1, 2, 3, 4, 5, 6, 7, 12, 13], 2)
            elbows = get_keypoint(x, [8, 9], 2)
            wrists = get_keypoint(x, [10, 11], 2)
            knees = get_keypoint(x, [14, 15], 2)
            ankles = get_keypoint(x, [16, 17], 2)

            torso = rearrange(torso, "b f (x y) -> b f x y", x= 3, y =6)       ## shape: b f 6 3
            torso = F.pad(input=torso, pad=(0, 0, 2, 1), mode='constant', value=0)   ## Shape: b f 6 6 
            torso = torso.unsqueeze(1)                                         ## Shape: b 1 f 6 6

            elbows = rearrange(elbows, "b f (x y) -> b f x y", x= 2, y =2)     ## Shape: b f 2 2 
            elbows = F.pad(input=elbows, pad=(2, 2, 2, 2), mode='constant', value=0) ## Shape: b f 6 6
            elbows = elbows.unsqueeze(1)

            wrists = rearrange(wrists, "b f (x y) -> b f x y", x= 2, y =2)     ## Shape: b f 2 2
            wrists = F.pad(input=wrists, pad=(2, 2, 2, 2), mode='constant', value=0) ## Shape: b f 6 6
            wrists = wrists.unsqueeze(1)                                       ## Shape: b 1 f 6 6

            knees = rearrange(knees, "b f (x y) -> b f x y", x= 2, y =2)       ## Shape: b f 2 2
            knees = F.pad(input=knees, pad=(2, 2, 2, 2), mode='constant', value=0) ## Shape: b f 6 6
            knees = knees.unsqueeze(1)                                         ## Shape: b 1 f 6 6

            ankles = rearrange(ankles, "b f (x y) -> b f x y", x= 2, y =2)     ## Shape: b f 4 2
            ankles = F.pad(input=ankles, pad=(2, 2, 2, 2), mode='constant', value=0) ## Shape: b f 6 6
            ankles = ankles.unsqueeze(1)                                       ## Shape: b 1 f 6 6

            torso_embed = self.torso_conv(torso)
            elbows_embed = self.elbows_conv(elbows)
            wrists_embed = self.wrists_conv(wrists)
            knees_embed = self.knees_conv(knees)
            ankles_embed = self.ankles_conv(ankles)

            # print(torso_embed.shape)
            # print(elbows_embed.shape)

            torso_embed = torch.flatten(torso_embed, start_dim=2)
            elbows_embed = torch.flatten(elbows_embed, start_dim=2)
            wrists_embed = torch.flatten(wrists_embed, start_dim=2)
            knees_embed = torch.flatten(knees_embed, start_dim=2)
            ankles_embed = torch.flatten(ankles_embed, start_dim=2)

            ## CONCAT THE RESULTS OF CONVOLUTION (N x 32 dimensional)
            torso_embed = rearrange(torso_embed, "b e n -> b (n e)")
            elbows_embed = rearrange(elbows_embed, "b e n -> b (n e)")
            wrists_embed = rearrange(wrists_embed, "b e n -> b (n e)")
            knees_embed = rearrange(knees_embed, "b e n -> b (n e)")
            ankles_embed = rearrange(ankles_embed, "b e n -> b (n e)")

            # print(torso_embed.shape)
            # print(elbows_embed.shape)

            x = torch.stack((torso_embed, elbows_embed, wrists_embed, knees_embed, ankles_embed), dim=1)

            return x
            

    def forward_features(self, x):
        
        # x = rearrange(x, 'b f (h w c) -> b c f h w', h=5, w=5, c=2)
        x = self.tubelet_embedding(x)
        # x = self.c3d(x)

        # x = rearrange(x, 'b o_c d h w -> b (d h w) o_c')
        # x = self.embedding(x)

        cls_token = self.cls_token.expand(x.shape[0], -1, -1)  # stole cls_tokens impl from Phil Wang, thanks
        # print(x.shape)
        # print(cls_token.shape)
        x = torch.cat((cls_token, x), dim=1)

        x = self.pos_drop(x + self.pos_embed)

        for blk in self.blocks:
            x = blk(x)

        x = self.norm(x)
        cls_token_final = x[:, 0]
        return cls_token_final
    
    def forward(self, x):
        x = self.forward_features(x)
        x = self.head(x)
        x = F.log_softmax(x, dim=1)
        return x

class TubeletTemporalPart_mean_chan_2_Transformer(nn.Module):
    '''
    BPFormer-V3
    '''
    def __init__(self, dataset=None, num_classes=13, num_frames=12, num_joints=17, in_chans=2, embed_dim=64, kernel=None, stride=None, depth=4,
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
        print('embed_dim', embed_dim)
        print('in_chans', in_chans)
        print('num_joints', num_joints)

        self.dataset = dataset
        
        if "NTU" in dataset or "HRC" in dataset:
            self.torso_conv = torch.nn.Conv3d(in_chans, embed_dim, kernel_size=kernel, stride=stride)
            self.elbows_conv = torch.nn.Conv3d(in_chans, embed_dim, kernel_size=kernel, stride=stride)
            self.wrists_conv = torch.nn.Conv3d(in_chans, embed_dim, kernel_size=kernel, stride=stride)
            self.knees_conv = torch.nn.Conv3d(in_chans, embed_dim, kernel_size=kernel, stride=stride)
            self.ankles_conv = torch.nn.Conv3d(in_chans, embed_dim, kernel_size=kernel, stride=stride)

        ### patch embedding
        # self.embedding = nn.Linear(num_joints*in_chans, embed_dim)

        ### Additional class token
        self.cls_token = nn.Parameter(torch.zeros(1, embed_dim))

        ### positional embedding including class token
        # num_pos = int(num_frames/5) * int(5/2) * int(5/2)
        
        self.pos_embed = nn.Parameter(torch.zeros(5+1, embed_dim)) ## 5 body parts + 1 pos embed

        self.pos_drop = nn.Dropout(p=drop_rate)

        self.blocks = nn.ModuleList([
            Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, dropout=dropout
                )
            for i in range(depth)])
        

        self.norm = nn.LayerNorm(embed_dim, eps=1e-6)

         # Representation layer
        '''if representation_size and not distilled:
            self.num_features = representation_size
            self.pre_logits = nn.Sequential(OrderedDict([
                ('fc', nn.Linear(embed_dim, representation_size)),
                ('act', nn.Tanh())
            ]))
        else:
            self.pre_logits = nn.Identity()'''
        #self.pre_logits = nn.Identity()

        # Classifier head(s)
        "Define standard linear + softmax generation step."
        "use learned linear transformation and softmax function to convert the output to predicted class probabilities"
        self.head = nn.Linear(embed_dim, num_classes) #no softmax is used
        
        # initialize weights
        self.init_weights()

        # taken from https://pytorch.org/tutorials/beginner/transformer_tutorial.html
    def init_weights(self):
          initrange = 0.1
        #   self.embedding.weight.data.uniform_(-initrange, initrange)
          self.head.bias.data.zero_()
          self.head.weight.data.uniform_(-initrange, initrange)

    def tubelet_embedding(self, x):
        if self.dataset == "NTU_3D":
            torso = get_keypoint(x, [], 3)
        elif self.dataset == "NTU_2D":
            torso = get_keypoint(x, [4, 3, 9, 21, 5, 2, 17, 1, 13], 2)
            elbows = get_keypoint(x, [10, 6], 2)
            wrists = get_keypoint(x, [11, 12, 24, 25, 7, 8, 22, 23], 2)
            knees = get_keypoint(x, [18, 14], 2)
            ankles = get_keypoint(x, [19, 20, 15, 16], 2)

            torso = rearrange(torso, "b f (x y c) -> b c f x y", c = 2, x= 3, y =3)       ## shape: b 2 f 3 3
            # torso = F.pad(input=torso, pad=(0, 0, 2, 1), mode='constant', value=0)   ## Shape: b f 6 6 
            # torso = torso.unsqueeze(1)                                         ## Shape: b 1 f 6 6

            elbows = rearrange(elbows, "b f (x y c) -> b c f x y", c = 2, x = 1, y = 2)     ## Shape: b 2 f 1 2 
            elbows = F.pad(input=elbows, pad=(1, 0, 1, 1), mode='constant', value=0) ## Shape : b 2 f 2 2
            # elbows = elbows.unsqueeze(1)

            wrists = F.pad(wrists ,(0,2)) # Extend wrists by one keypoint so that we can transform it into 3x3 
            wrists = rearrange(wrists, "b f (x y c) -> b c f x y", x= 3, y =3)     ## Shape: b f 3 3
            # wrists = rearrange(wrists, "b f (x y c) -> b c f x y", c = 2, x= 2, y = 4)     ## Shape: b 2 f 2 4
            # wrists = F.pad(input=wrists, pad=(0, 0, 1, 1), mode='constant', value=0)   ## Shape: b f 4 4
            # wrists = wrists.unsqueeze(1)                                      

            knees = rearrange(knees, "b f (x y c) -> b c f x y", c = 2, x= 1, y =2)       ## Shape: b 2 f 1 2
            knees = F.pad(input=knees, pad=(1, 0, 1, 1), mode='constant', value=0) ## Shape: b 2 f 2 2
            # knees = knees.unsqueeze(1)                                         

            ankles = rearrange(ankles, "b f (x y c) -> b c f x y", c = 2, x= 2, y =2)     ## Shape: b 2 f 2 2
            ankles = F.pad(input=ankles, pad=(1, 0, 1, 0), mode='constant', value=0) 
            # ankles = ankles.unsqueeze(1)                                       

            torso_embed = self.torso_conv(torso)
            elbows_embed = self.elbows_conv(elbows)
            wrists_embed = self.wrists_conv(wrists)
            knees_embed = self.knees_conv(knees)
            ankles_embed = self.ankles_conv(ankles)

            # OUTPUT SHAPE : b e n1 n2 n3

            # print(torso_embed.shape)
            # print(elbows_embed.shape)

            torso_embed = torch.flatten(torso_embed, start_dim=2)
            elbows_embed = torch.flatten(elbows_embed, start_dim=2)
            wrists_embed = torch.flatten(wrists_embed, start_dim=2)
            knees_embed = torch.flatten(knees_embed, start_dim=2)
            ankles_embed = torch.flatten(ankles_embed, start_dim=2)

            # OUTPUT SHAPE : b e (n1xn2xn3)


            torso_embed = torch.mean(torso_embed, dim=2)
            elbows_embed = torch.mean(elbows_embed, dim=2)
            wrists_embed = torch.mean(wrists_embed, dim=2)
            knees_embed = torch.mean(knees_embed, dim=2)
            ankles_embed = torch.mean(ankles_embed, dim=2)

            # OUTPUT SHAPE : b e

            # print(torso_embed.shape)
            # print(elbows_embed.shape)

            x = torch.stack((torso_embed, elbows_embed, wrists_embed, knees_embed, ankles_embed), dim=1)

            return x
        elif self.dataset == "HRC":
            torso = get_keypoint(x, [1, 2, 3, 4, 5, 6, 7, 12, 13], 2)
            elbows = get_keypoint(x, [8, 9], 2)
            wrists = get_keypoint(x, [10, 11], 2)
            knees = get_keypoint(x, [14, 15], 2)
            ankles = get_keypoint(x, [16, 17], 2)

            torso = rearrange(torso, "b f (x y c) -> b c f x y", c = 2, x= 3, y =3)       ## shape: b 2 f 3 3
            # torso = F.pad(input=torso, pad=(0, 0, 2, 1), mode='constant', value=0)   ## Shape: b 2 f 3 3 
            # torso = torso.unsqueeze(1)                                         

            elbows = rearrange(elbows, "b f (x y c) -> b c f x y", c = 2, x= 1, y =2)     ## Shape: b 2 f 1 2 
            elbows = F.pad(input=elbows, pad=(1, 0, 1, 1), mode='constant', value=0) ## Shape: b 2 f 2 2 
            # elbows = elbows.unsqueeze(1)

            wrists = rearrange(wrists, "b f (x y c) -> b c f x y", c = 2, x= 1, y =2)     ## Shape: b 2 f 1 2 
            wrists = F.pad(input=wrists, pad=(1, 0, 1, 1), mode='constant', value=0) ## Shape: b 2 f 2 2 
            # wrists = wrists.unsqueeze(1)                                       

            knees = rearrange(knees, "b f (x y c) -> b c f x y", c = 2, x= 1, y =2)       ## Shape: b 2 f 1 2
            knees = F.pad(input=knees, pad=(1, 0, 1, 1), mode='constant', value=0) ## Shape: b 2 f 2 2
            # knees = knees.unsqueeze(1)                                         

            ankles = rearrange(ankles, "b f (x y c) -> b c f x y", c = 2, x= 1, y =2)     ## Shape: b 2 f 1 2
            ankles = F.pad(input=ankles, pad=(1, 0, 1, 1), mode='constant', value=0) ## Shape: b 2 f 2 2
            # ankles = ankles.unsqueeze(1)                                       

            torso_embed = self.torso_conv(torso)
            elbows_embed = self.elbows_conv(elbows)
            wrists_embed = self.wrists_conv(wrists)
            knees_embed = self.knees_conv(knees)
            ankles_embed = self.ankles_conv(ankles)

            # print(torso_embed.shape)
            # print(elbows_embed.shape)

            torso_embed = torch.flatten(torso_embed, start_dim=2)
            elbows_embed = torch.flatten(elbows_embed, start_dim=2)
            wrists_embed = torch.flatten(wrists_embed, start_dim=2)
            knees_embed = torch.flatten(knees_embed, start_dim=2)
            ankles_embed = torch.flatten(ankles_embed, start_dim=2)


            torso_embed = torch.mean(torso_embed, dim=2)
            elbows_embed = torch.mean(elbows_embed, dim=2)
            wrists_embed = torch.mean(wrists_embed, dim=2)
            knees_embed = torch.mean(knees_embed, dim=2)
            ankles_embed = torch.mean(ankles_embed, dim=2)

            # print(torso_embed.shape)
            # print(elbows_embed.shape)

            x = torch.stack((torso_embed, elbows_embed, wrists_embed, knees_embed, ankles_embed), dim=1)

            return x
            

    def forward_features(self, x):
        
        # x = rearrange(x, 'b f (h w c) -> b c f h w', h=5, w=5, c=2)
        x = self.tubelet_embedding(x)
        # x = self.c3d(x)

        # x = rearrange(x, 'b o_c d h w -> b (d h w) o_c')
        # x = self.embedding(x)

        cls_token = self.cls_token.expand(x.shape[0], -1, -1)  # stole cls_tokens impl from Phil Wang, thanks
        # print(x.shape)
        # print(cls_token.shape)
        x = torch.cat((cls_token, x), dim=1)

        x = self.pos_drop(x + self.pos_embed)

        for blk in self.blocks:
            x = blk(x)

        x = self.norm(x)
        cls_token_final = x[:, 0]
        return cls_token_final
    
    def forward(self, x):
        x = self.forward_features(x)
        x = self.head(x)
        x = F.log_softmax(x, dim=1)
        return x

class TubeletTemporalPart_concat_chan_2_Transformer(nn.Module):
    '''
    BPFormer-V4
    '''
    ## Instead of mean, concatenate with a smaller dimension. 
    ## Also means you have to pad to make same dimension as largest input square. 
    ## Otherwise, the embeddings will have different lengths after concatenating.
    ## Consider two channels of the keypoints instead of flattening (x, y)
    ## Max
    def __init__(self, dataset=None, num_classes=13, num_frames=12, num_joints=17, in_chans=2, embed_dim=64, kernel=None, stride=None, depth=4,
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
        print('embed_dim', embed_dim)
        print('in_chans', in_chans)
        print('num_joints', num_joints)

        self.dataset = dataset
        
        if "NTU" in dataset or "HRC" in dataset:
            self.torso_conv = torch.nn.Conv3d(in_chans, embed_dim, kernel_size=kernel, stride=stride)
            self.elbows_conv = torch.nn.Conv3d(in_chans, embed_dim, kernel_size=kernel, stride=stride)
            self.wrists_conv = torch.nn.Conv3d(in_chans, embed_dim, kernel_size=kernel, stride=stride)
            self.knees_conv = torch.nn.Conv3d(in_chans, embed_dim, kernel_size=kernel, stride=stride)
            self.ankles_conv = torch.nn.Conv3d(in_chans, embed_dim, kernel_size=kernel, stride=stride)

        ### patch embedding
        # self.embedding = nn.Linear(num_joints*in_chans, embed_dim)

        num_embed = (int((num_frames-kernel[0])/stride[0]) + 1) * (int((3-kernel[1])/stride[1]) + 1) * (int((3-kernel[2])/stride[2]) + 1)
        
        self.final_embed_dim = num_embed * embed_dim
        ### Additional class token
        self.cls_token = nn.Parameter(torch.zeros(1, self.final_embed_dim))

        ### positional embedding including class token
        # num_pos = int(num_frames/5) * int(5/2) * int(5/2)

        
        
        

        self.pos_embed = nn.Parameter(torch.zeros(5+1, self.final_embed_dim)) ## 5 body parts + 1 pos embed

        self.pos_drop = nn.Dropout(p=drop_rate)

        self.blocks = nn.ModuleList([
            Block(
                dim=self.final_embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, dropout=dropout
                )
            for i in range(depth)])
        

        self.norm = nn.LayerNorm(self.final_embed_dim, eps=1e-6)

         # Representation layer
        '''if representation_size and not distilled:
            self.num_features = representation_size
            self.pre_logits = nn.Sequential(OrderedDict([
                ('fc', nn.Linear(embed_dim, representation_size)),
                ('act', nn.Tanh())
            ]))
        else:
            self.pre_logits = nn.Identity()'''
        #self.pre_logits = nn.Identity()

        # Classifier head(s)
        "Define standard linear + softmax generation step."
        "use learned linear transformation and softmax function to convert the output to predicted class probabilities"
        self.head = nn.Linear(self.final_embed_dim, num_classes) #no softmax is used
        
        # initialize weights
        self.init_weights()

        # taken from https://pytorch.org/tutorials/beginner/transformer_tutorial.html
    def init_weights(self):
          initrange = 0.1
        #   self.embedding.weight.data.uniform_(-initrange, initrange)
          self.head.bias.data.zero_()
          self.head.weight.data.uniform_(-initrange, initrange)

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
            elbows = F.pad(input=elbows, pad=(1, 0, 1, 1), mode='constant', value=0) ## Shape: b f 2 2
            # elbows = elbows.unsqueeze(1)

            wrists = F.pad(wrists ,(0,2)) # Extend wrists by one keypoint so that we can transform it into 3x3 
            wrists = rearrange(wrists, "b f (x y c) -> b c f x y", x= 3, y =3)     ## Shape: b f 3 3
            # wrists = F.pad(input=wrists, pad=(0, 0, 1, 1), mode='constant', value=0) ## Shape: b f 4 4
            # wrists = wrists.unsqueeze(1)                                       ## Shape: b 1 f 6 6

            knees = rearrange(knees, "b f (x y c) -> b c f x y", x= 1, y =2)       ## Shape: b f 1 2
            knees = F.pad(input=knees, pad=(1, 0, 1, 1), mode='constant', value=0) ## Shape: b f 2 2
            # knees = knees.unsqueeze(1)                                         ## Shape: b 1 f 6 6

            ankles = rearrange(ankles, "b f (x y c) -> b c f x y", x= 2, y =2)     ## Shape: b f 2 2
            ankles = F.pad(input=ankles, pad=(1, 0, 1, 0), mode='constant', value=0) ## Shape: b f 6 6
            # ankles = ankles.unsqueeze(1)                                       ## Shape: b 1 f 6 6

            torso_embed = self.torso_conv(torso)
            elbows_embed = self.elbows_conv(elbows)
            wrists_embed = self.wrists_conv(wrists)
            knees_embed = self.knees_conv(knees)
            ankles_embed = self.ankles_conv(ankles)

            # print(torso_embed.shape)
            # print(elbows_embed.shape)

            torso_embed = torch.flatten(torso_embed, start_dim=2)
            elbows_embed = torch.flatten(elbows_embed, start_dim=2)
            wrists_embed = torch.flatten(wrists_embed, start_dim=2)
            knees_embed = torch.flatten(knees_embed, start_dim=2)
            ankles_embed = torch.flatten(ankles_embed, start_dim=2)

            ## CONCAT THE RESULTS OF CONVOLUTION (N x 32 dimensional)
            torso_embed = rearrange(torso_embed, "b e n -> b (n e)")
            elbows_embed = rearrange(elbows_embed, "b e n -> b (n e)")
            wrists_embed = rearrange(wrists_embed, "b e n -> b (n e)")
            knees_embed = rearrange(knees_embed, "b e n -> b (n e)")
            ankles_embed = rearrange(ankles_embed, "b e n -> b (n e)")

            # print(torso_embed.shape)
            # print(elbows_embed.shape)

            x = torch.stack((torso_embed, elbows_embed, wrists_embed, knees_embed, ankles_embed), dim=1)

            return x
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
            elbows = F.pad(input=elbows, pad=(1, 0, 1, 1), mode='constant', value=0) ## Shape: b f 6 6
            # elbows = elbows.unsqueeze(1)

            wrists = rearrange(wrists, "b f (x y c) -> b c f x y", x= 1, y =2)     ## Shape: b 2 f 1 2
            wrists = F.pad(input=wrists, pad=(1, 0, 1, 1), mode='constant', value=0) ## Shape: b f 6 6
            # wrists = wrists.unsqueeze(1)                                       ## Shape: b 1 f 6 6

            knees = rearrange(knees, "b f (x y c) -> b c f x y", x= 1, y =2)       ## Shape: b 2 f 1 2
            knees = F.pad(input=knees, pad=(1, 0, 1, 1), mode='constant', value=0) ## Shape: b f 6 6
            # knees = knees.unsqueeze(1)                                         ## Shape: b 1 f 6 6

            ankles = rearrange(ankles, "b f (x y c) -> b c f x y", x= 1, y =2)     ## Shape: b 2 f 1 2
            ankles = F.pad(input=ankles, pad=(1, 0, 1, 1), mode='constant', value=0) ## Shape: b f 6 6
            # ankles = ankles.unsqueeze(1)                                       ## Shape: b 1 f 6 6

            torso_embed = self.torso_conv(torso)
            elbows_embed = self.elbows_conv(elbows)
            wrists_embed = self.wrists_conv(wrists)
            knees_embed = self.knees_conv(knees)
            ankles_embed = self.ankles_conv(ankles)

            # print(torso_embed.shape)
            # print(elbows_embed.shape)

            torso_embed = torch.flatten(torso_embed, start_dim=2)
            elbows_embed = torch.flatten(elbows_embed, start_dim=2)
            wrists_embed = torch.flatten(wrists_embed, start_dim=2)
            knees_embed = torch.flatten(knees_embed, start_dim=2)
            ankles_embed = torch.flatten(ankles_embed, start_dim=2)

            ## CONCAT THE RESULTS OF CONVOLUTION (N x 32 dimensional)
            torso_embed = rearrange(torso_embed, "b e n -> b (n e)")
            elbows_embed = rearrange(elbows_embed, "b e n -> b (n e)")
            wrists_embed = rearrange(wrists_embed, "b e n -> b (n e)")
            knees_embed = rearrange(knees_embed, "b e n -> b (n e)")
            ankles_embed = rearrange(ankles_embed, "b e n -> b (n e)")

            # print(torso_embed.shape)
            # print(elbows_embed.shape)

            x = torch.stack((torso_embed, elbows_embed, wrists_embed, knees_embed, ankles_embed), dim=1)

            return x
            

    def forward_features(self, x):
        
        # x = rearrange(x, 'b f (h w c) -> b c f h w', h=5, w=5, c=2)
        x = self.tubelet_embedding(x)
        # x = self.c3d(x)

        # x = rearrange(x, 'b o_c d h w -> b (d h w) o_c')
        # x = self.embedding(x)

        cls_token = self.cls_token.expand(x.shape[0], -1, -1)  # stole cls_tokens impl from Phil Wang, thanks
        # print(x.shape)
        # print(cls_token.shape)
        x = torch.cat((cls_token, x), dim=1)

        x = self.pos_drop(x + self.pos_embed)

        for blk in self.blocks:
            x = blk(x)

        x = self.norm(x)
        cls_token_final = x[:, 0]
        return cls_token_final
    
    def forward(self, x):
        x = self.forward_features(x)
        x = self.head(x)
        x = F.log_softmax(x, dim=1)
        return x


class TubeletTemporalSpatialPart_concat_chan_2_Transformer(nn.Module):
    '''
    BPTubeFormer Transformer
    '''
    def __init__(self, dataset=None, num_classes=13, num_frames=12, num_joints=17, in_chans=2, embed_dim_ratio=64, kernel=None, stride=None, depth=4,
                 num_heads=8, mlp_ratio=2., qkv_bias=True, qk_scale=None,
                 drop_rate=0., attn_drop_rate=0., dropout=0.2, pad_mode='constant',include_top=False,embed_dim_final=32,vit_model=None):
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
            include_top (boolean): Requires the final classification layer, False means just purely features
        """
        super().__init__()
        self.include_top=include_top
        self.in_chans = in_chans
        self.dataset = dataset
        self.pad_mode = pad_mode
        self.num_embed = (int((num_frames-kernel[0])/stride[0]) + 1) * (int((3-kernel[1])/stride[1]) + 1) * (int((3-kernel[2])/stride[2]) + 1)
        self.embed_dim_final = embed_dim_final
        self.vit_model = vit_model

        embed_dim = embed_dim_ratio * self.num_embed #* 5   #### temporal embed_dim is embed_dim_ratio x num_embed (op of 3dconv) x 5 (no. of body parts)
        # print(embed_dim)
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
        if self.vit_model: 
            print(f"Embedding Dimension :- {embed_dim}")
            print(f"Final Embedding Dimension :- {self.embed_dim_final}")
            print(f"Total Embedding Dimension :- {embed_dim+self.embed_dim_final}")
            self.embed = nn.Linear(embed_dim+self.embed_dim_final, self.embed_dim_final)
            self.head_2 = nn.Linear(self.embed_dim_final,num_classes)
        else:
            self.embed = nn.Linear(embed_dim, self.embed_dim_final)
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
        # print(torso_embed.size())
        elbows_embed = self.Elbow_forward_features(elbows)
        # print(elbows_embed.size())
        wrists_embed = self.Wrist_forward_features(wrists)
        # print(wrists_embed.size())

        knees_embed = self.Knee_forward_features(knees)
        # print(knees_embed.size())
        ankles_embed = self.Ankle_forward_features(ankles)
        # print(ankles_embed.size())
        x = torch.stack((torso_embed, elbows_embed, wrists_embed, knees_embed, ankles_embed), dim=1)
        print(f"after stacking :{x.shape}")
        return x

    def forward(self, x,video_frames):


        x = self.tubelet_embedding(x)
        print(f"After tublete embedding {x.shape}")
        x = self.forward_features(x) 
        print(f"After forward features shape {x.shape}")

        
        #Early fusion

        if self.include_top:
            x = self.head(x)
            x = F.log_softmax(x, dim=1)
            return x
        else:
            if self.vit_model:#early fusion
                # print("Running early fusion")
                context_frame = self.vit_model(video_frames)
                # print(context_frame.shape)
                early_fusion = torch.concat([x,context_frame],axis=1)
                # print(early_fusion.shape)
                fusion = self.embed(early_fusion)
                # final_result = self.head_2(fusion)
                return fusion
        

"""
Aditya
"""

class vit_model(nn.Module):
    """
        Uses PyTorch implementation of VIT
        Params
        =========
        in_channel -> input of the final layer(generally 768, default ViT layer)
        out_channel -> output of the final layer(for our case will be the embedding size so we can align to the last)
        fine_tune_mode -> If enabled freezes all the layer except the last

    """
    def __init__(self,embed_dim,in_channel,out_channel,fine_tune_mode=False,pretrained_model="vit16",weights="IMAGENET1K_V1"):
        super().__init__()
        # self.vit_model = vit_b_16()
        # self.vit_model.heads.head = nn.Linear(in_channel,out_channel)
        if not fine_tune_mode: #If fine tune mode is off
            self.vit_model = VisionTransformer(image_size= 224,
                patch_size= 16,
                num_layers= 4,
                num_heads= 3,
                hidden_dim= 120,
                mlp_dim= 40,
                dropout=   0.0,
                attention_dropout=   0.0,
                num_classes=  embed_dim,
            )
        
            print(f"Loaded ViT model, ViT embeding dim : {out_channel} Fine tune mode : {fine_tune_mode}")
        else:
            if pretrained_model=="vit16": #with 16 patches
                imagenet_weights =  ViT_B_16_Weights[weights]
                self.vit_model = vit_b_16(imagenet_weights)
                for i in self.vit_model.parameters():
                    i.requires_grad=False
                
                self.vit_model.heads.head = nn.Linear(768,embed_dim)
                getTrainableParameters = list(map(lambda x: x.requires_grad, self.vit_model.parameters()))[-1]
                print(f"Loaded ViT model, ViT embeding dim : {out_channel} Fine tune mode : {fine_tune_mode}")
                print(f"Last layer trainable : {getTrainableParameters}")

            else: #with 32 patches
                imagenet_weights= ViT_B_32_Weights[weights]
                self.vit_model = vit_b_32(imagenet_weights)
                for i in self.vit_model.parameters():
                    i.requires_grad=False
                self.vit_model.heads.head = nn.Linear(768,embed_dim)
                getTrainableParameters = list(map(lambda x: x.requires_grad, self.vit_model.parameters()))[-1]
                print(f"Loaded ViT model, ViT embeding dim : {out_channel} Fine tune mode : {fine_tune_mode}")
                print(f"Last layer trainable : {getTrainableParameters}")

    def forward(self,x):
        x = self.vit_model(x)
        return x

"""
Takes in 2 models and learn parameters of these two embedding
"""


class ensemble(nn.Module): #takes in tublete embedding
    def __init__(self,model_1,model_2,input_dim=32,output_dim=13,device=None):
        super().__init__()
        self.model_1 = model_1 #Tublete embedding
        
        print(f"Loaded model 1")
        self.model_2 = model_2 #would be VIT or LSTM
        print(f"Loaded model 2")
        # print(f"Loaded model 2 : In cuda {next(self.model_1.parameters()).is_cuda}")
        self.fc_layer = nn.Linear(input_dim,output_dim) #to change later
        print("Loaded Ensemble")
    
    def forward(self,coordinates,frames): #coordinate input, frames input
        # print(coordinates.shape)
        # print(frames.shape)
        start = time.time()
        

        coordinates = self.model_1(coordinates,frames)
        # print(f"Model 1 : {abs(start-time.time())}")
        # print("Pass 1 ")
        # print(coordinates.shape)
        # print("check fram shape")
        # print(frames.shape)
        # start = time.time()
        frames = self.model_2(frames)
        # print(f"Model 2 : {abs(start-time.time())}")
        # print("Pass 2 ")
        # print(frames.shape)
        fusion = torch.cat((coordinates,frames),dim=1)
       
        # print(fusion.shape)
        x = self.fc_layer(fusion)
        x= F.log_softmax(x,dim=1)
        # print("Processed")
        return x








# class ResNet(nn.Module):
#     def __init__(self,embed_dim):
#         super().__init__()
#         self.resnet = resnet18(ResNet18_Weights)
#         print("Loaded ResNet model")
#         for i in self.resnet.parameters():
#             i.requires_grad=False
#         self.layer = nn.Linear(512,embed_dim)
#         self.resnet.fc = self.layer

#     def forward(self,x):
#         return self.resnet(x)


#Cross attention implementation

class CrossAttention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.,return_attention=True):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        # NOTE scale factor can be manually set to be compat with prev weights
        self.scale = qk_scale or head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.cross_attn_scores = None

        self.project_skeleton = nn.Linear(dim,dim)
        self.project_context_key = nn.Linear(dim,dim)
        self.project_context_value = nn.Linear(dim,dim)
        
    def forward(self, encoder_1,encoder_2): 
        # print(encoder_1.shape)
        # print(encoder_2.shape)

        B, N,C = encoder_1.shape #Batch, segment length, embed dim
        # print(B,N,C)
        q, k, v = encoder_1, encoder_2, encoder_2   # make torchscript happy (cannot use tensor as tuple)

        q = self.project_skeleton(q).reshape(B,N,self.num_heads,C//self.num_heads).permute(0,2,1,3) #batch,segment length, num_heads,embed dim/num_head
        k = self.project_context_key(k).reshape(B,1,self.num_heads,C//self.num_heads).permute(0,2,1,3) #same
        # print(f"Query : {q.shape} ")
        # print(f"Key : {k.shape} ")
        attention = q@k.transpose(-2,-1)
        attention_weights = nn.Softmax(dim=2)(attention)
        self.cross_attn_scores=attention_weights
        # print(attention_weights.size())
        v = self.project_context_value(v).reshape(B,1,self.num_heads,C//self.num_heads).permute(0,2,1,3)
        final = (attention_weights@v).reshape(B, N, C)

        return final
        # print(final.size())
        # print(k)
        # return
class CrossAttentionBlock(nn.Module):

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 dropout=0., act_layer=nn.GELU):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim, eps=1e-6)
        self.attn = CrossAttention(dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        #self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.dropout = nn.Dropout(dropout) #first try a simple dropout instead of drop path
        self.norm2 = nn.LayerNorm(dim, eps=1e-6)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
        
    def forward(self, encoder_1,encoder_2):
        #architecture taken from these paper https://arxiv.org/pdf/2204.04564.pdf

        x = self.dropout(self.attn(encoder_1,encoder_2))

        x = self.norm2(encoder_1+x)
        x = self.mlp(x)
        return x

    
# class CrossAttention(nn.Module):
#     def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.,return_attention=True):
#         super().__init__()
#         self.num_heads = num_heads
#         head_dim = dim // num_heads
#         # NOTE scale factor can be manually set to be compat with prev weights
#         self.scale = qk_scale or head_dim ** -0.5
#         self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
#         self.attn_drop = nn.Dropout(attn_drop)
#         self.proj = nn.Linear(dim, dim)
#         self.proj_drop = nn.Dropout(proj_drop)
#         self.cross_attn_scores = None

#         self.project_skeleton = nn.Linear(dim,dim)
#         self.project_context_key = nn.Linear(dim,dim)
#         self.project_context_value = nn.Linear(dim,dim)


#     def forward(self, encoder_1,encoder_2): 
#         #encoder_1 skeleton trajectories, encoder_2 visual information
#         #value = output of model which contains skeleton trajectories, #query and key model which consists of spatial information
#         # print(encoder_1.shape)
#         # print(encoder_2.shape)
#         B, N,C = encoder_1.shape #Batch, segment length, embed dim
      
#         q, k, v = encoder_1, encoder_2, encoder_2   # make torchscript happy (cannot use tensor as tuple)
#         #torch.Size([1, 100, 64])
        
#         q = self.project_skeleton(q).reshape(B,N,self.num_heads,C//self.num_heads).permute(0,2,1,3) #batch,segment length, num_heads,embed dim/num_head

#         k = self.project_context_key(k).reshape(B,1,self.num_heads,C//self.num_heads).permute(0,2,1,3) #same
#         v = self.project_context_value(k).reshape(B,1,self.num_heads,C//self.num_heads).permute(0,2,1,3)

#         # print("reshapoed key ", k.size())
#         # print(q.size())
#         # B,N,C = encoder_2.shape
#         # print(v.size())
#         # v = v.squeeze()

#         attn = (q @ k.transpose(-2, -1)) * self.scale  # [100, 4, 21, 1]) [100, 4, segment legnth, 1 for each context]) 
#         # print(q@k.transpose(-2,-1))
#         attn = attn.softmax(dim=-2)
   
#         self.cross_attn_scores = attn
#         attn = self.attn_drop(attn)
    

#         x = ((attn @ v)).reshape(B, N, C)

            
#         x = self.proj(x)
        
#         x = self.proj_drop(x)

#         return x

# class CrossAttentionBlock(nn.Module):

#     def __init__(self,dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
#                  dropout=0., act_layer=nn.GELU):
#         super().__init__()
#         self.norm1 = nn.LayerNorm(dim, eps=1e-6)
#         self.attn = CrossAttention(dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
#         # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
#         #self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
#         self.dropout = nn.Dropout(dropout) #first try a simple dropout instead of drop path
#         self.norm2 = nn.LayerNorm(dim, eps=1e-6)
#         mlp_hidden_dim = int(dim * mlp_ratio)
#         self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
        
#     def forward(self, encoder_1,encoder_2):
#         #architecture taken from these paper https://arxiv.org/pdf/2204.04564.pdf

#         x = self.dropout(self.attn(encoder_1,encoder_2))
#         x = self.norm2(encoder_2.permute(1,0,2) +x)
#         x = self.mlp(x)
#         return x
    


    

##########################################################################




#Fusion model that inherits temporal transformer

class FusionModel_TemporalTransformer(TemporalTransformer):
    def __init__(self,*args,fusion_type="early",cross_attention=True,**kwargs):
        # prin
        # print(True if fusion_type!="el" else False)

        if (fusion_type=="el") or (fusion_type=="l"):
            self.embed_fusion = kwargs["embed_dim"]//2
        else:
            self.embed_fusion = kwargs["embed_dim"] 
        self.fusion=fusion_type
        print(f"Fusion type : '{self.fusion}'")
        # self.embed_fusion = kwargs["embed_dim"] if (fusion_type!="el" or fusion_type!="l") else kwargs["embed_dim"]//2 #Embedding dimesnion is split so that, 64//2 -> 32 for ViT 32 for Temporal transformer

        kwargs["embed_dim"]= self.embed_fusion
        super().__init__(*args, **kwargs)
                
        self.model_2 = VisionTransformer(image_size= 224,
                patch_size= 16,
                num_layers= 4,
                num_heads= 6,
                hidden_dim= 60,
                mlp_dim= 40,
                dropout=   0.0,
                attention_dropout=   0.0,
                num_classes=  self.embed_fusion,
            )

        if (fusion_type=="el") or (fusion_type=="l") :
            self.head = nn.Linear(self.embed_fusion*2,kwargs["num_classes"])
        else:
            self.head = nn.Linear(self.embed_fusion,kwargs["num_classes"])

        self.need_crossattention = cross_attention
        if self.need_crossattention:
            self.cross_attention_block =  nn.ModuleList([
            CrossAttentionBlock(
                dim=kwargs["embed_dim"], num_heads=kwargs["num_heads"], mlp_ratio=kwargs["mlp_ratio"], qkv_bias=kwargs["qkv_bias"], qk_scale=kwargs["qk_scale"],
                drop=kwargs["drop_rate"], attn_drop=kwargs["attn_drop_rate"], dropout=kwargs["dropout"]
                 #drop_path=dpr[i] #first try a simple dropout instead of drop path
                )
            for i in range(kwargs["depth"])])
            
            # CrossAttentionBlock(
            #   self.embed_fusion, 4, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
            #      dropout=0., act_layer=nn.GELU
            #      #drop_path=dpr[i] #first try a simple dropout instead of drop path
            #     ) 
            # self.cross_attention_block =  nn.ModuleList([CrossAttentionBlock(
            #   self.embed_fusion, 4, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
            #      dropout=0., act_layer=nn.GELU
            #      #drop_path=dpr[i] #first try a simple dropout instead of drop path
            #     )     for i in range(4)])

        # self.head = nn.Linear(self.embed_fusion if (fusion_type!="el") or (fusion_type!="l") else self.embed_fusion*2,kwargs["num_classes"])
        self.fusion_block = None #can be a nn.Sequential() to do the fusion.
        self.classification_block = None
        if self.fusion!=None:
            if self.fusion!="l":
                self.pos_embed = nn.Parameter(torch.zeros(kwargs["num_frames"]+2, kwargs["embed_dim"]))
            else:
                self.pos_embed = nn.Parameter(torch.zeros(kwargs["num_frames"]+1, kwargs["embed_dim"]))
        else:
            self.pos_embed = nn.Parameter(torch.zeros(kwargs["num_frames"]+1, kwargs["embed_dim"]))
        # self.pos_embed = nn.Parameter(torch.zeros(kwargs["num_frames"]+2, kwargs["embed_dim"])) if self.fusion!=None or self.fusion \
        #                                 else nn.Parameter(torch.zeros(kwargs["num_frames"]+1, kwargs["embed_dim"]))  
        self.model_2_output = None 
    def forward_features(self, x,image=None):
        #print("call forward features")
        # print(f"x shape: {x.shape}")

        x = self.embedding(x)
        # print("this")
        # print(x.size())
        if self.fusion!=None:
            img = self.model_2(image).unsqueeze(0)
            if (self.fusion=="el" or self.fusion=="l"):
                self.model_2_output=img
        else:
            if self.need_crossattention:
                img = self.model_2(image).unsqueeze(0)
                self.model_2_output=img
        # print(img.size())
        # print("s")
        #print(f"self cls_token shape: {self.cls_token.shape}")

        #print("expand cls_token")
        cls_token = self.cls_token.expand(x.shape[0], -1, -1)  # stole cls_tokens impl from Phil Wang, thanks

        #print(f"expanded cls_token shape: {cls_token.shape}")
        #print(f"embedded x shape: {x.shape}")

        x = torch.cat((cls_token, x), dim=1)

        # print(x.size())
        if self.fusion is not None:
            if self.fusion!="l":
                img = torch.permute(img,(1,0,2))
                x=torch.cat((x,img),dim=1)
    
        # print(x.size())
        #print(f"embedded x + cls_token shape: {x.shape}")

        #print(f"pos_embed shape: {self.pos_embed.shape}")
        #print(f"x + self.pos_embed shape: {(x + self.pos_embed).shape}")

        x = self.pos_drop(x + self.pos_embed)
        # print(f"pos_drop x shape: {x.shape}")

        #x = self.blocks(x)
        for blk in self.blocks:
            x = blk(x)
            #print(f"blocks(x) shape: {x.shape}")

        x = self.norm(x)
        # print("x shape:", x.shape)
        

        if self.need_crossattention:
            for cross_attention in self.cross_attention_block:
                x = cross_attention(x,self.model_2_output)    
        cls_token_final = x[:, 0]     
        # if self.need_crossattention:
        #     for i in self.cross_attention_block:
        #         cls_token_final = i(cls_token_final,self.model_2_output)
        #     return cls_token_final
        #return self.pre_logits(x[:, 0])
        return cls_token_final

    def forward(self,x,image=None):
        if self.fusion=="e": #Early fusion temporal transformer
            x = self.forward_features(x,image)
            x = self.head(x)
            x = F.log_softmax(x, dim=1)
            #do early fusion here
            return x
        if self.fusion=="el": #Early & late fusion transformer
            x = self.forward_features(x,image)
            # print(torch.permute(self.model_2_output,(1,0,2)).size())
            x = torch.concat([x,self.model_2_output.squeeze()],dim=1)
            x = self.head(x)
            x = F.log_softmax(x, dim=1)
            return x

        if self.fusion=="l": #late fusion transformer
            x = self.forward_features(x,image)
            # print(x.size())
            x = torch.concat([x,self.model_2_output.squeeze()],dim=1)
            x = self.head(x)
            x = F.log_softmax(x, dim=1)
            return x

        if self.fusion==None: #Regular temporal transformer
            x = self.forward_features(x,image)
            x = self.head(x)
            x = F.log_softmax(x, dim=1)
            return x






class TemporalTransformerFusion(nn.Module):
    def __init__(self, fusion_type="l",cross_attention=False,model_2=None,num_classes=13, num_frames=12, num_joints=17, in_chans=2, embed_dim=64, depth=4,
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
        print('embed_dim', embed_dim)
        print('in_chans', in_chans)
        print('num_joints', num_joints)

        if (fusion_type=="el") or (fusion_type=="l"):
            self.embed_fusion = embed_dim//2
        else:
            self.embed_fusion = embed_dim 
        self.fusion=fusion_type
        self.model_2 = model_2
        ### patch embedding
        self.embedding = nn.Linear(num_joints*in_chans, embed_dim)
        self.need_crossattention = cross_attention
        ### Additional class token
        self.cls_token = nn.Parameter(torch.zeros(1, embed_dim))

        ### positional embedding including class token
        if self.fusion!=None:
            if self.fusion!="l":
                self.pos_embed = nn.Parameter(torch.zeros(num_frames+2, embed_dim))
            else:
                self.pos_embed = nn.Parameter(torch.zeros(num_frames+1, embed_dim))
        else:
            self.pos_embed = nn.Parameter(torch.zeros(num_frames+1, embed_dim))

        self.pos_drop = nn.Dropout(p=drop_rate)

        #dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule

        self.blocks = nn.ModuleList([
            Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, dropout=dropout
                 #drop_path=dpr[i] #first try a simple dropout instead of drop path
                )
            for i in range(depth)])
        

        self.norm = nn.LayerNorm(embed_dim, eps=1e-6)


        #self.pre_logits = nn.Identity()

        # Classifier head(s)
        "Define standard linear + softmax generation step."
        "use learned linear transformation and softmax function to convert the output to predicted class probabilities"
        if (fusion_type=="el") or (fusion_type=="l") :
            self.head = nn.Linear(self.embed_fusion*2+self.embed_fusion,num_classes)
        else:
            self.head = nn.Linear(self.embed_fusion,num_classes)
        
        # initialize weights
        self.init_weights()

        # taken from https://pytorch.org/tutorials/beginner/transformer_tutorial.html
    def init_weights(self):
          initrange = 0.1
          self.embedding.weight.data.uniform_(-initrange, initrange)
          self.head.bias.data.zero_()
          self.head.weight.data.uniform_(-initrange, initrange)

    def forward_features(self, x,image=None):
        #print("call forward features")
        # print(f"x shape: {x.shape}")

        x = self.embedding(x)
        # print("this")
        # print(x.size())
        if self.fusion!=None:
            if self.fusion!="l":
                img = self.model_2(image).unsqueeze(0)
            # if (self.fusion=="el" or self.fusion=="l"):
            #     self.model_2_output=img
        else:
            if self.need_crossattention:
                img = self.model_2(image).unsqueeze(0)
                self.model_2_output=img
        # print(img.size())
        # print("s")
        #print(f"self cls_token shape: {self.cls_token.shape}")

        #print("expand cls_token")
        cls_token = self.cls_token.expand(x.shape[0], -1, -1)  # stole cls_tokens impl from Phil Wang, thanks

        #print(f"expanded cls_token shape: {cls_token.shape}")
        #print(f"embedded x shape: {x.shape}")

        x = torch.cat((cls_token, x), dim=1)

        # print(x.size())
        if self.fusion is not None:
            if self.fusion!="l":
                img = torch.permute(img,(1,0,2))
                x=torch.cat((x,img),dim=1)
    
        # print(x.size())
        #print(f"embedded x + cls_token shape: {x.shape}")

        #print(f"pos_embed shape: {self.pos_embed.shape}")
        #print(f"x + self.pos_embed shape: {(x + self.pos_embed).shape}")

        x = self.pos_drop(x + self.pos_embed)
        # print(f"pos_drop x shape: {x.shape}")

        #x = self.blocks(x)
        for blk in self.blocks:
            x = blk(x)
            #print(f"blocks(x) shape: {x.shape}")

        x = self.norm(x)
        # print("x shape:", x.shape)
        

        if self.need_crossattention:
            for cross_attention in self.cross_attention_block:
                x = cross_attention(x,self.model_2_output)    
        cls_token_final = x[:, 0]     
        # if self.need_crossattention:
        #     for i in self.cross_attention_block:
        #         cls_token_final = i(cls_token_final,self.model_2_output)
        #     return cls_token_final
        #return self.pre_logits(x[:, 0])
        return cls_token_final
    
    def forward(self,x,image=None):
        if self.fusion=="e": #Early fusion temporal transformer
            x = self.forward_features(x,image)
            x = self.head(x)
            x = F.log_softmax(x, dim=1)
            #do early fusion here
            return x
        if self.fusion=="el": #Early & late fusion transformer
            x = self.forward_features(x,image)
            # print(torch.permute(self.model_2_output,(1,0,2)).size())
            x = torch.concat([x,self.model_2_output.squeeze()],dim=1)
            x = self.head(x)
            x = F.log_softmax(x, dim=1)
            return x

        if self.fusion=="l": #late fusion transformer
            x = self.forward_features(x)
            print(x.size())
            img = self.model_2(image).unsqueeze(0)
            # print(x.size())
            x = torch.concat([x,img.squeeze()],dim=1)
            x = self.head(x)
            x = F.log_softmax(x, dim=1)
            return x

        if self.fusion==None: #Regular temporal transformer
            x = self.forward_features(x,image)
            x = self.head(x)
            x = F.log_softmax(x, dim=1)
            return x
        


"""
This is to test if the numpy memmap is worth it
"""


class TemporalTransformer_nmap(TemporalTransformer):
    def __init__(self, *args,**kwargs):
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
        super().__init__(*args, **kwargs)

        # self.embedding = nn.Linear(num_joints*in_chans, embed_dim//2)

        self.align_dimension = nn.Linear(768,kwargs["embed_dim"]) # this is for the frames
        self.pos_embed = nn.Parameter(torch.zeros(kwargs["num_frames"]+2, kwargs["embed_dim"]))
        # initialize weights
        self.init_weights()

        # taken from https://pytorch.org/tutorials/beginner/transformer_tutorial.html
    def init_weights(self):
          initrange = 0.1
          self.embedding.weight.data.uniform_(-initrange, initrange)
          self.head.bias.data.zero_()
          self.head.weight.data.uniform_(-initrange, initrange)

    def forward_features(self, x,frames):

        #print("call forward features")
        # print(f"x shape: {x.shape}")

        x = self.embedding(x)
        # print(x.size())
        descriptor_ = self.align_dimension(frames)
        # print(f"After aligning {descriptor_.shape}" )
        descriptor_ = descriptor_.unsqueeze(1)
        # print(f"After unsqueezing {descriptor_.shape}")

        # print(descriptor_.size())
        # print(f"Joint size : {x.shape}")
        # print(f"Descriptor size : {descriptor_.shape} ")
        x = torch.cat((x,descriptor_),dim=1)
        # print(x.shape)        


        
        #print(f"self cls_token shape: {self.cls_token.shape}")

        #print("expand cls_token")
        cls_token = self.cls_token.expand(x.shape[0], -1, -1)  # stole cls_tokens impl from Phil Wang, thanks
        # print(f"Cls token {cls_token.shape}")
        #print(f"expanded cls_token shape: {cls_token.shape}")
        #print(f"embedded x shape: {x.shape}")

        x = torch.cat((cls_token, x), dim=1)
        # print(x.shape)
        #print(f"embedded x + cls_token shape: {x.shape}")

        #print(f"pos_embed shape: {self.pos_embed.shape}")
        #print(f"x + self.pos_embed shape: {(x + self.pos_embed).shape}")

        x = self.pos_drop(x + self.pos_embed)
        # print(f"pos_drop x shape: {x.shape}")

        #x = self.blocks(x)
        for blk in self.blocks:
            x = blk(x)
            #print(f"blocks(x) shape: {x.shape}")

        x = self.norm(x)
        # print("x shape:", x.shape)
        cls_token_final = x[:, 0]
        # print(cls_token_final.size())
        #return self.pre_logits(x[:, 0])
        return cls_token_final
    

    def forward(self, x,frames):
        x = self.forward_features(x,frames)
        
        x = self.head(x)
    
        x = F.log_softmax(x, dim=1)
        return x



"""
late fusion only
"""

class TemporalTransformer_nmap_lfusion(TemporalTransformer):
    def __init__(self, *args,**kwargs):
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
        super().__init__(*args, **kwargs)

        # self.embedding = nn.Linear(num_joints*in_chans, embed_dim//2)

        self.align_dimension = nn.Linear(768,kwargs["embed_dim"]) # this is for the frames
        self.head = nn.Linear(kwargs["embed_dim"]*2,kwargs["num_classes"])
        # self.pos_embed = nn.Parameter(torch.zeros(kwargs["num_frames"]+2, kwargs["embed_dim"]))
        # initialize weights
        self.init_weights()
        self.conv_1d= nn.Conv1d(in_channels=1, out_channels=kwargs["embed_dim"], kernel_size=5, stride=1)
        self.global_avgpool = nn.AdaptiveAvgPool1d(1)
        # taken from https://pytorch.org/tutorials/beginner/transformer_tutorial.html
    def init_weights(self):
          initrange = 0.1
          self.embedding.weight.data.uniform_(-initrange, initrange)
          self.head.bias.data.zero_()
          self.head.weight.data.uniform_(-initrange, initrange)

    def forward_features(self, x):

        #print("call forward features")
        # print(f"x shape: {x.shape}")

        x = self.embedding(x)
        # print(x.size())
        # descriptor_ = self.align_dimension(frames)
        # print(f"After aligning {descriptor_.shape}" )
        # descriptor_ = descriptor_.unsqueeze(1)
        # print(f"After unsqueezing {descriptor_.shape}")

        # print(descriptor_.size())
        # print(f"Joint size : {x.shape}")
        # print(f"Descriptor size : {descriptor_.shape} ")
        # x = torch.cat((x,descriptor_),dim=1)
        # print(x.shape)        


        
        #print(f"self cls_token shape: {self.cls_token.shape}")

        #print("expand cls_token")
        cls_token = self.cls_token.expand(x.shape[0], -1, -1)  # stole cls_tokens impl from Phil Wang, thanks
        # print(f"Cls token {cls_token.shape}")
        #print(f"expanded cls_token shape: {cls_token.shape}")
        #print(f"embedded x shape: {x.shape}")

        x = torch.cat((cls_token, x), dim=1)
        # print(x.shape)
        #print(f"embedded x + cls_token shape: {x.shape}")

        #print(f"pos_embed shape: {self.pos_embed.shape}")
        #print(f"x + self.pos_embed shape: {(x + self.pos_embed).shape}")

        x = self.pos_drop(x + self.pos_embed)
        # print(f"pos_drop x shape: {x.shape}")

        #x = self.blocks(x)
        for blk in self.blocks:
            x = blk(x)
            #print(f"blocks(x) shape: {x.shape}")

        x = self.norm(x)
        # print("x shape:", x.shape)
        cls_token_final = x[:, 0]
        # print(cls_token_final.size())
        #return self.pre_logits(x[:, 0])
        return cls_token_final
    

    def forward(self, x,frames):
        x = self.forward_features(x)
        frames = self.align_dimension(frames)
        # frames = frames.unsqueeze(0).view(-1,1,768) #to reshape 1 channel
        
        # frames = self.conv1d(frames) 
        # frames = self.global_avgpool(frames).squeeze(2) #because it generates (batch,64,1)

        fusion = torch.concat([x,frames],dim=1)
        x = self.head(fusion)
    
        x = F.log_softmax(x, dim=1)
        return x


"""

Temporal only for cross attention

"""



class TemporalTransformer_CrossAttention(TemporalTransformer):
    def __init__(self, *args,**kwargs):
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
        super().__init__(*args, **kwargs)

        # self.embedding = nn.Linear(num_joints*in_chans, embed_dim//2)

        self.align_dimension = nn.Linear(768,kwargs["embed_dim"]) # this is for the frames
        self.head = nn.Linear(kwargs["embed_dim"],kwargs["num_classes"])
        # self.pos_embed = nn.Parameter(torch.zeros(kwargs["num_frames"]+2, kwargs["embed_dim"]))
        # initialize weights
        self.init_weights()

        self.cross_attention_block = nn.ModuleList([
            CrossAttentionBlock(
                dim=64, num_heads=4
                )
            for i in range(3)])
        # taken from https://pytorch.org/tutorials/beginner/transformer_tutorial.html
    def init_weights(self):
          initrange = 0.1
          self.embedding.weight.data.uniform_(-initrange, initrange)
          self.head.bias.data.zero_()
          self.head.weight.data.uniform_(-initrange, initrange)

    def forward_features(self, x):

        #print("call forward features")
        # print(f"x shape: {x.shape}")

        x = self.embedding(x)

        #print(f"self cls_token shape: {self.cls_token.shape}")

        #print("expand cls_token")
        cls_token = self.cls_token.expand(x.shape[0], -1, -1)  # stole cls_tokens impl from Phil Wang, thanks
        # print(f"Cls token {cls_token.shape}")
        #print(f"expanded cls_token shape: {cls_token.shape}")
        #print(f"embedded x shape: {x.shape}")

        x = torch.cat((cls_token, x), dim=1)
        # print(x.shape)
        #print(f"embedded x + cls_token shape: {x.shape}")

        #print(f"pos_embed shape: {self.pos_embed.shape}")
        #print(f"x + self.pos_embed shape: {(x + self.pos_embed).shape}")

        x = self.pos_drop(x + self.pos_embed)
        # print(f"pos_drop x shape: {x.shape}")

        #x = self.blocks(x)
        for blk in self.blocks:
            x = blk(x)
            #print(f"blocks(x) shape: {x.shape}")

        x = self.norm(x)

        return x
    

    def forward(self, x,frames):
        x = self.forward_features(x)

        frames = self.align_dimension(frames)
        
        for i in self.cross_attention_block:
            x = i(x,frames)
        x = self.head(x)
        x = x[:,0]
        x = F.log_softmax(x, dim=1)
        return x