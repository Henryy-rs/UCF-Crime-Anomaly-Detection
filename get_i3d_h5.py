import os
import h5py
import numpy as np
import json


def get_h5():
    feat_npy_root = FEATURES_DIR   # need to be replaced
    feat_files = os.listdir(feat_npy_root)
    feat_files = [item for item in feat_files if item.endswith('.npy')]

    feat_dict = {}

    print('Start ...')
    count = 0
    for item in feat_files:
        #added
        video_name = item.replace('-', '/').split('.')[0] +".mp4"
        print(video_name)
        filepath = os.path.join(feat_npy_root, item)
        feat = np.load(filepath)
        
        feat_dict[video_name] = feat

        count += 1
        if count%1000 == 0:
            print('Processed %d files.'%count)

    print('Processed %d files.'%count)


    print('Writing file ...')

    fid = h5py.File(OUTPUT_FILE_NAME, 'w')

    for vid in feat_dict.keys():
        if vid in fid:
            print('WARNING: group name exists.')
            continue

        fid.create_group(vid).create_dataset('i3d_rgb_features', data=feat_dict[vid])

    print('Done.')


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--FEATURES_DIR', dest='FEATURES_DIR', type=str, default='./temp')
    parser.add_argument('--OUTPUT_FILE_NAME', dest='OUTPUT_FILE_NAME', type=str, default='./temp')
    args = parser.parse_args()
    params = vars(args)

    VIDEO_DIR = params['VIDEO_DIR']
    VIDEO_NAME = params['OUTPUT_FILE_NAME']
    get_h5()
