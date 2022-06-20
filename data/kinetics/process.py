#
#
#

import argparse
import csv
import os
import json
import numpy as np
from numpy.lib.format import open_memmap

MIN_FRAMES = 10
MAX_INTERLEAVE = 50
NOISE_THRESHOLD = 0.8
SCORE_THRESHOLD = 2.5
VALID_THRESHOLD = 0.7

NUM_CHANNELS = 3
NUM_FRAMES = 300
NUM_PEOPLE = 2
NUM_JOINTS = None

def parse_args():
    parser = argparse.ArgumentParser('preprocess kinetics skeleton')
    parser.add_argument('--labels', type=str, help='cs file containing (id, name) for each label')
    parser.add_argument('--input', type=str, help='directory containing the input files')
    parser.add_argument('--output', type=str, help='directory where to save the output')
    parser.add_argument('--dataset', type=str, help='the dataset used for pose estimation (posetrack | coco)')
    
    args = parser.parse_args()
    return args

def set_joints(dataset: str):
    global NUM_JOINTS
    if dataset == 'posetrack':
        NUM_JOINTS = 17
    elif dataset == 'coco':
        NUM_JOINTS = 18
    else: 
        raise ValueError(f'{dataset} is not a valid dataset')

def get_labels_dict(file) -> dict:
    labels = {}
    with open(file, 'r') as f:
        data = csv.reader(f, delimiter=',')
        for idx, row in enumerate(data): 
            if idx == 0:
                continue
            labels[row[1]] = row[0]
    
    return labels

def get_bodies(frames: 'list[dict]') -> dict:
    bodies = {}
    for idx, frame in enumerate(frames):
        assert idx + 1 == frame['frame_id'], 'Error in frame index'
        
        for person in frame['people']:
            body_id = person['track_id']
            bbox = np.array(person['bbox'], dtype=np.float32)
            keypoints = np.array(person['keypoints'], dtype=np.float32)
            
            if body_id not in bodies:
                body = {}
                body['bbox'] = np.array([bbox])
                body['keypoints'] = np.array([keypoints])
                body['frames'] = [idx]
                bodies[body_id] = body
            else:
                body = bodies[body_id]
                body['bbox'] = np.vstack((body['bbox'], [bbox]))
                body['keypoints'] = np.vstack((body['keypoints'], [keypoints]))
                body['frames'].append(idx)
    
    return bodies

def get_valid_frames(body: dict) -> list: 
    valid_frames = []
    for idx, joints in enumerate(body['keypoints']):
        # denoise by score
        #bscore = bbox[4]
        jscore_mean = np.mean(joints[:, 2])
        jscore_var = np.var(joints[:, 2])
        if jscore_mean < 0.5 and jscore_var > 0.3: 
            continue
        
        # denoise by spread
        x = joints[:, 0]
        y = joints[:, 1]
        if (x.max() - x.min()) > NOISE_THRESHOLD * (y.max() - y.min()):
            continue
        
        valid_frames.append(idx)
    
    return valid_frames
    

def get_valid_bodies(bodies: dict) -> 'list[tuple[str, dict]]':
    motions = []
    
    for (id, body) in bodies.items(): 
        frames = body['frames']
        valid_frames = get_valid_frames(body)
        
        if len(valid_frames) / len(frames) < VALID_THRESHOLD: 
            continue
        
        if len(valid_frames) < MIN_FRAMES:
            continue
        
        diff = np.diff(valid_frames)
        if np.any(diff > MAX_INTERLEAVE): 
            continue
        
        body['frames'] = [body['frames'][i] for i in valid_frames]
        body['bbox'] = body['bbox'][valid_frames]
        body['keypoints'] = body['keypoints'][valid_frames]
        
        motion = np.sum(np.var(body['keypoints'], axis=0)[0:2])
        motions.append((id, motion))
    
    motions = sorted(motions, key=lambda x: x[1], reverse=True)
    valid_bodies = []
    for (id, _) in motions:
        valid_bodies.append((id, bodies[id]))
        
    return valid_bodies

def get_body_center(keypoints: np.array, dataset: str) -> np.ndarray: 
    if dataset == 'posetrack':
        pos = [1, 11, 12]
    elif dataset == 'coco':
        pos = [1, 8, 11]
    else: 
        raise ValueError(f'{dataset} is not a valid dataset')
    
    i = 0
    center = None
    for i in range(len(keypoints)):
        joints = [keypoints[i][p] for p in pos]
        tmp = np.mean(joints, axis = 0)
            
        if (center is None) or (tmp[2] > center[2]):
            center = tmp

    return center

def translate_sequence(skels: 'list[np.ndarray]', dataset: str) -> 'list[np.ndarray]':
    if len(skels) == 0: 
        return []
    
    center = get_body_center(skels[0], dataset)
    
    for joints in skels:
        joints[:, :, 0:2] -= center[0:2]
    
    return skels
    
def process_sequence(sequence: dict, dataset: str) -> any: 
    bodies = get_bodies(sequence['frames'])
    valid_bodies = get_valid_bodies(bodies)
    
    if len(valid_bodies) == 0:
        return None
    
    pskels = [body['keypoints'] for (_, body) in valid_bodies[:NUM_PEOPLE]]
    pskels = translate_sequence(pskels, dataset)
    
    skels = np.zeros((NUM_CHANNELS, NUM_FRAMES, NUM_JOINTS, NUM_PEOPLE))
    for idx, (_, body) in enumerate(valid_bodies[:NUM_PEOPLE]): 
        frames = np.array(body['frames'])
        valid_frames = frames < NUM_FRAMES
        frames_idx = frames[valid_frames]
        
        skels[:, frames_idx, :, idx] = pskels[idx][valid_frames].transpose((0, 2, 1))
    
    return skels

def main(): 
    args = parse_args()
    set_joints(args.dataset)
    
    labels = get_labels_dict(args.labels)
    num_labels = len(labels)
    
    files = os.listdir(args.input)
    print(f'Found {len(files)} sequences to preprocess')
    
    tmp_output_file = os.path.join(args.output, 'tmp.memmap')
    output_file = os.path.join(args.output, 'kinetics.memmap')
    tmp_out_skels = open_memmap(
        tmp_output_file,
        dtype=np.float32,
        mode='w+',
        # N x C x T x V x M
        shape=(len(files), NUM_CHANNELS, NUM_FRAMES, NUM_JOINTS, NUM_PEOPLE)
    )
    
    tmp_out_labels = np.zeros((len(files), num_labels), dtype=np.int8)
    
    current_idx = 0
    for idx, file_name in enumerate(files):
        print(f'({idx}) Preprocessing sequence {os.path.splitext(file_name)[0]}')
        file = os.path.join(args.input, file_name)
        assert os.path.exists(file), f'File ${file} does not exist'
        
        with open(file, 'rb') as f:
            data = json.load(f)
            
        skels = process_sequence(data, args.dataset)
        if skels is None:
            print(f'({idx}) Sequence {os.path.splitext(file_name)[0]} not valid')
            continue
        
        tmp_out_skels[current_idx] = skels
        
        label = data['label']
        tmp_out_labels[current_idx][int(labels[label])] = 1
        current_idx += 1
    
    out_skels = tmp_out_skels
    out_labels = tmp_out_labels
    if current_idx == len(files):
        # all files were correct
        os.rename(tmp_output_file, output_file)
    else: 
        out_skels = open_memmap(
            output_file,
            dtype=np.float32,
            mode='w+',
            shape=(current_idx, NUM_CHANNELS, NUM_FRAMES, NUM_JOINTS, NUM_PEOPLE)
        )
        
        out_labels = np.zeros((current_idx, num_labels), dtype=np.int8)
        
        out_skels[:] = tmp_out_skels[:current_idx]
        out_labels[:] = tmp_out_labels[:current_idx]
        del tmp_out_skels
        os.remove(tmp_output_file)
    
    out = os.path.join(args.output, f'kinetics{num_labels}.npz')
    np.savez(out, x=out_skels, y=out_labels)
    
    os.remove(output_file)
    
    print(f'Found {current_idx} valid sequences')
        
if __name__ == '__main__':
    main()