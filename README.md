# ACTION CLASSIFICATION

## NTU Dataset

The NTU Dataset contains video data as well as skeleton data.
### NTU Videos
The videos are stored in `/home/s2435462/HRC/NTU/Videos`
The data includes both NTURGB-D and NTURGB-D-120 datasets.
In total there are 114480 videos.
One example file name is `S018C001P042R002A120_rgb.avi`. There are 5 components in the filename.

* S018: Setup number
* C001: Camera ID
* P042: Performer (subject) ID
* R002: Replication count
* A120: Action class

For more details about these, refer [GitHub](https://github.com/shahroudy/NTURGB-D) and [Paper](https://www.cv-foundation.org/openaccess/content_cvpr_2016/papers/Shahroudy_NTU_RGBD_A_CVPR_2016_paper.pdf)

### NTU Skeleton
The skeleton data is stored in `/home/s2435462/HRC/NTU/skeleton/nturgb+d_skeletons`.
Some 535 videos do not have skeleton data and the video names are listed in `/home/s2435462/HRC/NTU/skeleton/NTU_RGBD120_samples_with_missing_skeletons.txt`.
The .skeleton files need to be converted and they are converted to .npy format using the script `txt2npy.py` script. This stores 3D and 2D keypoints in .npy format.
These .npy files are saved in `/home/s2435462/HRC/NTU/skeleton/nturgb+d_skeletons_npy/`
So in effect, we have 113945 skeleton files.

Now, these .npy files can be written in CSV format with 2D and 3D data stored separately. This is done using the `trajectory_csv.py` script. These are in:
* 2D: `/home/s2435462/HRC/NTU/skeleton/trajectory_csv_2D/`
* 3D: `/home/s2435462/HRC/NTU/skeleton/trajectory_csv_3D/`
with 143162 files each

The folder structure looks like this:

```
trajectory_csv_2D/
    A057/
        P029/
            S008C003P029R002A057_0.csv
            S008C003P029R002A057_1.csv
            S008C002P029R002A057_0.csv
            S008C002P029R002A057_1.csv
            .
            .
            .
        P007/
            S013C001P007R001A057_0.csv
            S013C001P007R001A057_1.csv
            S015C001P007R002A057_0.csv
            S015C001P007R002A057_1.csv
            .
            .
            .
    A100/
        P007/
            S013C001P007R001A100_0.csv
            S013C001P007R001A100_1.csv
            S015C001P007R002A100_0.csv
            S015C001P007R002A100_1.csv
            .
            .
            .

```
So the first level corresponds to the 120 different action classes and within each such folder, the next level corresponds to the person/subject number. In the last level, we have each CSV file which is in the format `0S008C003R002.csv`. Here, the first character represents the person inside the video (each video may contain more than one person) and the rest are same components as explained above. The same repeats for trajectory_csv_3D.

And for each file, we have:

```
0.0,1036.233,517.3242, .....
1.0,1036.252,518.9068, .....
2.0,1036.161,518.8845, .....
.
.
.
```
So the each line corresponds to one frame in the video and the first value is the frame number. The remaining values are the list of keypoints. So for 2D data, we would be having 2x25 values, while for 3D, we would have 3x25 values.

In total, we have 143162 trajectory files.

## PYTHON SCRIPTS

### `load_trajectories_NTU.py`


This script is used to load the trajectory CSV files into Trajectory classes, store them in a dict format and pickle save them in `.dat` format. You can set the dimension as either `2D` or `3D` and depending on this, the corresponding folder would be loaded. The data is saved in `trajectories_NTU_2D.dat` file in `/home/s2435462/HRC/data/`.

The dictionary looks like this:

```
{
    'S018C001P042R002A120_0' : <The Trajectory class>,
    'S018C001P042R002A120_1' : <The Trajectory class>,
    .
    .
}
```

### `prepare_trajectories_NTU.py`

This script is to remove short trajectories and prepare them into train and test datasets. It loads the pickled .dat files and splits them into 80% and 20%.

These are them saved as `trajectories_train_NTU_2D.dat` and `trajectories_test_NTU_2D.dat` in `/home/s2435462/HRC/data/`


We now have the test and train datasets saved!!

### `utils.py`

This file stores some utility functions like SetupLogger, printstatistics, etc..

### `train_transformer_cross_val_NTU.py`

This is the main training file.

It starts by setting up a logger and reading all arguments.

The train and test trajectories are then loaded from the output of `prepare_trajectories.py`.

Some short trajectories are then removed.

It uses K-fold cross validation for training.

The training results are stored in the format: 'fold', 'epoch', 'LR', 'Training Loss', 'Validation Loss', 'Validation Accuracy', 'Time'

The testing results are stored in the format : 'fold', 'label', 'video', 'person', 'prediction', 'log_likelihoods', 'logits'

The Trajectory datasets are then created using the TrajectoryDataset classes. It is initialized using the output of `extract_fixed_sized_segments()` which returns the segmented coordinates.

Then for each fold, a train and validation dataloader is defined. The model is also initialized. 

For all the epochs, the data from the train dataloader is passed to the model. The validation dataset is used for validation results. The training results are saved to a file.

If the patience is exceeded or the final epoch is done, the training is stopped. The test dataset is then used for test results.

### `trajectory.py`

### `transformer.py`

# RESULTS
### Kayleigh

|  **Model** | **c** | **f** | **BA** |
|:----------:|:-----:|:-----:|:------:|
| temporal_1 |  256  |   24  |  0.476 |
| temporal_2 |  128  |   12  | 0.4325 |
| temporal_3 |  256  |   12  | 0.4071 |
| temporal_4 |  128  |   12  | 0.3887 |
|     ST     |   32  |   60  | 0.4926 |
|    SBPT    |   16  |   60  | 0.4876 |

### AMJ

| **Dataset** |  **Model** | **c** | **f** | **BA** |
|:-----------:|:----------:|:-----:|:-----:|:------------:|
|    NTU_2D   | temporal_1 |  256  |   24  |    0.4617    |
|    NTU_2D   | temporal_2 |  128  |   12  |    0.4861    |
|    NTU_2D   | temporal_3 |  256  |   12  |    0.2465    |
|    NTU_2D   | temporal_4 |  128  |   12  |    0.3342    |
|    NTU_2D   |     ST     |   32  |   60  |    0.4288    |
|    NTU_2D   |    SBPT    |   16  |   60  |    0.4886    |

## PROGRESS

### 9th November 2022
* Got results for temporal_1, temporal_2, temporal_3 and temporal_4
* Currently running on ST and SBPT models (had to decrease batch size, 1 hour per epoch)
* Decomposed keypoints globally and locally (NTU_2D, NTU_3D and HRC)

#### Doubts
* Should ST and SBPT be trained fully? Or stop after one fold? Since it will take days
* Architecture for global and local keypoints?
* Local keypoints are normalized?
* 'Committee chair' and 'Committee member UT' in the proposal form?

### 28th October 2022
* RT Report Work    
* Ran NTU_2D on temporal_1/temporal_2/temporal_3
* Ran NTU_3D on temporal_1/
* Implemented Tensorboard for better visualization
* Implemented MLflow for better logging
* Revamped folder structures
* Used YAML configuration files instead of cmdline arguments
* Uses classes for the Trajectory dataset instead of plain lists
* Implemented git version control
* Added logging feature instead of just printing
* Cleaned the code

#### Doubts
* All models/ only ST (best performing Kayleigh)?
* Cross Validation necessary? Can I avoid?
* After getting ViTPose output, use that alone? or NTU default keypoints also? Or the best among them?
* FP registration delay


## TODO

* ~~Remove short trajectories? (ensure if it's atleast segment length)~~
* ~~Saving the best model? torch.save vs torch.state_dict~~
    * ~~Save inside current_loss << min_loss~~
* ~~train_loss vs val_loss~~
    * ~~loss vs loss.item()~~
    * ~~running_loss?~~
    * ~~Training accuracy~~
* ~~Balanced accuracy on folds?~~

* NTU120 2D and 3D on the existing models
* Tubelet
* Global, local decomposition
* ResNet


c 32 
c 16 7.5 minutes
c 8  5.9 minutes
c 4  5.0 minutes
c 2                 183510


c 2 b 100 killed
c 2 b 200 running

temporal_1 : /home/s2435462/HRC/results/NTU_2D/temporal_1_c32_b1000/logs/logs.log

temporal_2 : /home/s2435462/HRC/results/NTU_2D/temporal_2_c32_b1000/logs/logs.log

temporal_3 : /home/s2435462/HRC/results/NTU_2D/temporal_3_c32_b1000_2/logs/logs.log

temporal_4 : /home/s2435462/HRC/results/NTU_2D/temporal_4_c32_b1000/logs/logs.log

*spatio-temporal : /home/s2435462/HRC/results/NTU_2D/spatial-temporal_3_c32_b1000_6/logs/logs.log

*parts : /home/s2435462/HRC/results/NTU_2D/parts_3_c32_b100_8/logs/logs.log