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
            0S008C003R002.csv
            1S008C003R002.csv
            0S008C002R002.csv
            1S008C002R002.csv
            .
            .
            .
        P007/
            0S013C001R001.csv
            1S013C001R001.csv
            0S015C001R002.csv
            .
            .
            .
    A100/
            0S008C003R002.csv
            1S008C003R002.csv
            0S008C002R002.csv
            1S008C002R002.csv
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

This script is to prepare the trajectories into train and test datasets. It loads the pickled .dat files and splits them into 80% and 20%.

These are them saved as `trajectories_train_NTU_2D.dat` and `trajectories_test_NTU_2D.dat` in `/home/s2435462/HRC/data/`


We now have the test and train datasets saved!!

trajectory_id = category + '_' + folder_name + '_' + csv_file_name

A001_P001_0S006C002R002.csv

S018C001P042R002A120

0S006C002R002.csv

