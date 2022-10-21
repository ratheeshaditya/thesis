from statistics import mean
import logging
import sys

def SetupLogger(name):
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    ch = logging.StreamHandler(stream=sys.stdout)
    ch.setLevel(logging.INFO)

    formatter = logging.Formatter('%(asctime)s %(message)s')
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    return logger

def smaller_than_mean(lengths, mean):
    return len([x for x in lengths if x <= mean])

def print_statistics(train_crime_trajectories, test_crime_trajectories, logger):
    train_frame_lengths = []
    test_frame_lengths = []
    for key in train_crime_trajectories:
        #print(key, ' : ', train_crime_trajectories[key])
        num_of_frames = len(train_crime_trajectories[key])
        #print('number of frames:', num_of_frames)
        
        train_frame_lengths.append(num_of_frames)

    for key in test_crime_trajectories:
        num_of_frames = len(test_crime_trajectories[key])
        
        test_frame_lengths.append(num_of_frames)

    logger.info('TRAIN minimum: %d', min(train_frame_lengths))
    logger.info('TRAIN maximum: %d', max(train_frame_lengths))
    logger.info('TRAIN mean: %f', mean(train_frame_lengths))

    logger.info('TEST minimum: %d', min(test_frame_lengths))
    logger.info('TEST maximum: %d', max(test_frame_lengths))
    logger.info('TEST mean: %f', mean(test_frame_lengths))

    logger.info('TRAIN smaller_than_mean: %d', smaller_than_mean(train_frame_lengths, mean(train_frame_lengths)))
    logger.info('TEST smaller_than_mean: %d', smaller_than_mean(test_frame_lengths, mean(test_frame_lengths)))
