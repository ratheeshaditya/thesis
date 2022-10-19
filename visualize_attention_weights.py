import os
import matplotlib.pyplot as plt
import numpy as np
from einops import rearrange
import seaborn
seaborn.set_context(context="paper")
#seaborn.set(font_scale=2)

def draw_spatial(data, ax):
    seaborn.heatmap(data, 
                    square=True, vmin=0.0, vmax=1.0, cbar=False, ax=ax, cmap="mako")

def draw(data, ax):
    seaborn.heatmap(data, square=True, vmin=0.0, vmax=1.0, cbar=False, ax=ax)

#Visualize attention weights
def visualize_attention_weights(model_type, model, test_sample, test_index, image_location, filename):
    
    num_layers = len(model.blocks)
    num_heads = model.blocks[0].attn.num_heads
    
    #image_location = '/data/s3447707/MasterThesis/images_attention_weights/'
    
    layer = num_layers-1
    #layer = -1
    print("Encoder Layer", layer+1)
    
    sequence_length = len(model.blocks[0].attn.attn_scores[0,0,0,:].data)
    print('sequence_length', sequence_length)
    
    labelsize = 50
    
    if model_type=='temporal_2':
        step = 8
    else:
        step = 4
        
    
    for h in range(8):
        print('h', h)
        fig, axs = plt.subplots(1, 1, figsize=(10, 10))
        
        #draw(model.blocks[layer].attn.attn_scores[test_index,h].data, list(range(0,sequence_length)), list(range(0,sequence_length)), ax=axs)
        draw(model.blocks[layer].attn.attn_scores[test_index,h].data, ax=axs)
        
        image_name = test_sample + '_index_' + str(test_index) + '_layer_' + str(layer+1) + '_head_' + str(h+1) + '_' + model_type + '_transformer.jpg'
        image_name = os.path.join(image_location, image_name)
        
        labels = np.arange(0, sequence_length, step)
        #plt.setp(axs, xticks=labels+0.5, xticklabels=labels, yticks=labels+0.5, yticklabels=labels)
        plt.xticks(np.arange(0, sequence_length, step)+0.5, np.arange(0, sequence_length, step))
        plt.yticks(np.arange(0, sequence_length, step)+0.5, np.arange(0, sequence_length, step))
        
        axs.tick_params(axis='both', which='major', labelsize=labelsize)
        axs.set_xlim(0, sequence_length) #x-axis, 0 (left) to sequence_length (right)
        axs.set_ylim(sequence_length, 0) #y-axis, 0 (top) to sequence_length (bottom)
        axs.xaxis.set_ticks_position('top')
        
        #plt.show()
        plt.savefig(image_name)
        print('image saved to %s' %(image_name))
        
        #visualize spatial attention weights
        
        if model_type == 'spatial-temporal':
            num_frames = sequence_length-1
        
            spatial_scores = model.Spatial_blocks[layer].attn.attn_scores.data
            
            input_length = len(spatial_scores[0,0,0,:].data)
            print('input_length', input_length)
            
            #rearrange scores
            print('spatial_scores shape', spatial_scores.shape)
            spatial_scores = rearrange(spatial_scores, '(b f) h j1 j2  -> b f h j1 j2', f=num_frames) 
            print('spatial_scores shape', spatial_scores.shape)
                
            for frame in range(num_frames):
                fig_spatial, axs_spatial = plt.subplots(1, 1, figsize=(10, 10))
                
                draw_spatial(spatial_scores[test_index,frame,h].data, ax=axs_spatial) 
               
                image_name = test_sample + '_index_' + str(test_index) + '_layer_' + str(layer+1) + '_frame_' + str(frame+1) + '_head_' + str(h+1) + '_spatial_score_' + model_type + '_transformer.jpg'
                image_name = os.path.join(image_location, image_name)
                
                step = 4
                labels = np.arange(0, input_length, step)
                plt.xticks(np.arange(0, input_length, step)+0.5, np.arange(0, input_length, step))
                plt.yticks(np.arange(0, input_length, step)+0.5, np.arange(0, input_length, step))
                
                axs_spatial.tick_params(axis='both', which='major', labelsize=labelsize)
                axs_spatial.set_xlim(0, input_length) #x-axis, 0 (left) to input_length (right)
                axs_spatial.set_ylim(input_length, 0) #y-axis, 0 (top) to input_length (bottom)
                axs_spatial.xaxis.set_ticks_position('top')
                
                #plt.show()
                plt.savefig(image_name)
                print('image saved to %s' %(image_name))


        if model_type == 'parts':
            num_frames = sequence_length-1
            
            torso_scores = model.Torso_blocks[layer].attn.attn_scores.data
            elbow_scores = model.Elbow_blocks[layer].attn.attn_scores.data
            wrist_scores = model.Wrist_blocks[layer].attn.attn_scores.data
            knee_scores = model.Knee_blocks[layer].attn.attn_scores.data
            ankle_scores = model.Ankle_blocks[layer].attn.attn_scores.data
            
            part_labels = ['torso', 'elbows', 'wrists', 'knees', 'ankles']
            
            body_part_scores = [torso_scores, elbow_scores, wrist_scores, knee_scores, ankle_scores]
            
            for i in range(len(part_labels)):
                #print('part_labels[i]', part_labels[i])
                
                spatial_scores = body_part_scores[i]
                #rearrange scores
                print('spatial_scores shape', spatial_scores.shape)
                spatial_scores = rearrange(spatial_scores, '(b f) h j1 j2  -> b f h j1 j2', f=num_frames) 
                print('spatial_scores shape', spatial_scores.shape)
            
                for frame in range(num_frames):
                    fig_spatial, axs_spatial = plt.subplots(1, 1, figsize=(10, 10))
                    
                    input_length = len(spatial_scores[0,0,0,:].data)
                    print('input_length', input_length)
                    
                    draw_spatial(spatial_scores[test_index,frame,h].data, ax=axs_spatial) 
                   
                    image_name = test_sample + '_index_' + str(test_index) + '_layer_' + str(layer+1) + '_frame_' + str(frame+1) + '_head_' + str(h+1) + '_' + part_labels[i] + '_score_' + model_type + '_transformer.jpg'
                    image_name = os.path.join(image_location, image_name)
                    
                    if part_labels[i] == 'torso':
                        step = 2
                    else:
                        step = 1 #remaining body parts consist of only two joints
                    labels = np.arange(0, input_length, step)
                    plt.xticks(np.arange(0, input_length, step)+0.5, np.arange(0, input_length, step))
                    plt.yticks(np.arange(0, input_length, step)+0.5, np.arange(0, input_length, step))
                    
                    axs_spatial.tick_params(axis='both', which='major', labelsize=labelsize)
                    axs_spatial.set_xlim(0, input_length) #x-axis, 0 (left) to input_length (right)
                    axs_spatial.set_ylim(input_length, 0) #y-axis, 0 (top) to input_length (bottom)
                    axs_spatial.xaxis.set_ticks_position('top')
                    
                    #plt.show()
                    plt.savefig(image_name)
                    print('image saved to %s' %(image_name))
        

