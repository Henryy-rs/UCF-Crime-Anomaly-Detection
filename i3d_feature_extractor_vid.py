# coding: utf-8
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import json
import argparse
import os 
import time
import numpy as np
from PIL import Image

#
import tensorflow as tf

import i3d_model

_SAMPLE_VIDEO_FRAMES = 64
_IMAGE_SIZE = 224
_CHECKPOINT_PATHS = {
    'rgb': 'data/checkpoints/rgb_scratch/model.ckpt',
    'rgb600': 'data/checkpoints/rgb_scratch_kin600/model.ckpt',
    'flow': 'data/checkpoints/flow_scratch/model.ckpt',
    'rgb_imagenet': 'C:/deep/I3D-Feature-Extractor/data/checkpoints/rgb_imagenet/model.ckpt',
    'flow_imagenet': 'data/checkpoints/flow_imagenet/model.ckpt',
}
FLAGS = tf.flags.FLAGS

tf.flags.DEFINE_string('eval_type', 'joint', 'rgb, rgb600, flow, or joint')
tf.flags.DEFINE_boolean('imagenet_pretrained', True, '')

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
  # Restrict TensorFlow to only allocate 1GB of memory on the first GPU
  try:
    tf.config.experimental.set_virtual_device_configuration(
        gpus[0],
        [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=4096)])
    logical_gpus = tf.config.experimental.list_logical_devices('GPU')
    print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
  except RuntimeError as e:
    # Virtual devices must be set before GPUs have been initialized
    print(e)


def feature_extractor():
    imagenet_pretrained = FLAGS.imagenet_pretrained
    # loading net
    #net = i3d.InceptionI3d(00, spatial_squeeze=True, final_endpoint='Logits')
    #rgb_input = tf.placeholder(tf.float32, shape=(batch_size, _SAMPLE_VIDEO_FRAMES, _IMAGE_SIZE, _IMAGE_SIZE, 3))
    
    #_, end_points = net(rgb_input, is_training=False, dropout_keep_prob=1.0)
    #end_feature = end_points['avg_pool3d']
    rgb_input = tf.placeholder(
        tf.float32,
        shape=(1, _SAMPLE_VIDEO_FRAMES, _IMAGE_SIZE, _IMAGE_SIZE, 3))

    with tf.variable_scope('RGB'):
        rgb_model = i3d.InceptionI3d(
            400, spatial_squeeze=True, final_endpoint='Logits')
        rgb_logits, _ = rgb_model(
            rgb_input, is_training=False, dropout_keep_prob=1.0)
        rgb_logits, end_points = rgb_model(
            rgb_input, is_training=False, dropout_keep_prob=1.0)
        end_feature = end_points['avg_pool3d']
          
    sess = tf.Session()
    """
    rgb_variable_map = {}
    for variable in tf.compat.v1.global_variables():
        rgb_variable_map[variable.name.replace(':0', '')[len('inception_i3d/'):]] = variable
    #print(rgb_variable_map)
    saver = tf.compat.v1.train.Saver(var_list=rgb_variable_map)
    saver.restore(sess, _CHECKPOINT_PATHS['rgb_imagenet'])
    """
    rgb_variable_map = {}
    for variable in tf.global_variables():
        if variable.name.split('/')[0] == 'RGB':
            #rgb_variable_map[variable.name.replace(':0', '')[len('RGB/inception_i3d/'):]] = variable
            #else:
            rgb_variable_map[variable.name.replace(':0', '')] = variable
    rgb_saver = tf.train.Saver(var_list=rgb_variable_map, reshape=True)

    if imagenet_pretrained:
        rgb_saver.restore(sess, _CHECKPOINT_PATHS['rgb_imagenet'])
    else:
        rgb_saver.restore(sess, _CHECKPOINT_PATHS[eval_type])
    tf.logging.info('RGB checkpoint restored')
    #rgb_sample = np.load(_SAMPLE_PATHS['rgb'])
    #tf.logging.info('RGB data loaded, shape=%s', str(rgb_sample.shape))

    video_list = open(VIDEO_PATH_FILE).readlines()
    video_list = [name.strip() for name in video_list]
    print('video_list', video_list)
    if not os.path.isdir(OUTPUT_FEAT_DIR):
        os.makedirs(OUTPUT_FEAT_DIR)

    print('Total number of videos: %d'%len(video_list))
    
    for cnt, video_name in enumerate(video_list):
        video_path = os.path.join(VIDEO_DIR, video_name)
        feat_path = os.path.join(OUTPUT_FEAT_DIR, video_name + '.npy')

        if os.path.exists(feat_path):
            print('Feature file for video %s already exists.'%video_name)
            continue
        video_path = video_path.split(" ")[0]
        video_path = video_path.replace("\\", "/")
        temp_path = 'C:/deep/I3D-Feature-Extractor/temp'
        #jpg로 만들기
        ####################
        #video_path = os.path.join(VIDEO_DIR, video_name)
        #print('video_path', video_path)
        video_name = video_name.split(" ")[0] 
        frame_path = os.path.join(temp_path, video_name)
        frame_path = frame_path.replace("/", "\\")
        if not os.path.exists(frame_path):
            os.makedirs(frame_path)
		
        print('Extracting video frames ...')
        # using ffmpeg to extract video frames into a temporary folder
        # example: ffmpeg -i video_validation_0000051.mp4 -q:v 2 -f image2 output/image%5d.jpg
        os.system('ffmpeg -i ' + video_path + ' -q:v 2 -f image2 ' + frame_path + '/image_%5d.jpg')


        print('Extracting features ...')
        total_frames = len(os.listdir(frame_path))
        if total_frames == 0:
            error_fid.write(video_name+'\n')
            print('Fail to extract frames for video: %s'%video_name)
            continue
        #############################
        
        n_frame = len([ff for ff in os.listdir(frame_path) if ff.endswith('.jpg')])
        
        print('Total frames: %d'%n_frame)
        
        features = []

        n_feat = int(n_frame // 8)
        n_batch = n_feat // batch_size + 1
        print('n_frame: %d; n_feat: %d'%(n_frame, n_feat))
        print('n_batch: %d'%n_batch)

        for i in range(n_batch):
            input_blobs = []
            for j in range(batch_size):
                input_blob = []
                for k in range(L):
                    idx = i*batch_size*L + j*L + k
                    idx = int(idx)
                    idx = idx%n_frame + 1
                    #image = Image.open(os.path.join(frame_path, 'image_%5d.jpg'%idx))
                    image = Image.open(os.path.join(frame_path,  'image_{:05d}.jpg'.format(idx)))
                    image = image.resize((resize_w, resize_h))
                    image = np.array(image, dtype='float32')
                    
                    '''
                    image[:, :, 0] -= 104.
                    image[:, :, 1] -= 117.
                    image[:, :, 2] -= 123.
                    '''
                    image[:, :, :] -= 127.5
                    image[:, :, :] /= 127.5
                    input_blob.append(image)
                    #print(k)
                
                input_blob = np.array(input_blob, dtype='float32')
                
                input_blobs.append(input_blob)

            input_blobs = np.array(input_blobs, dtype='float32')
            print(i)

            clip_feature = sess.run(end_feature, feed_dict={rgb_input: input_blobs})
            clip_feature = np.reshape(clip_feature, (-1, clip_feature.shape[-1]))
            
            features.append(clip_feature)

        features = np.concatenate(features, axis=0)
        features = features[:n_feat:2]   # 16 frames per feature  (since 64-frame snippet corresponds to 8 features in I3D)
        
        #features = features.numpy()
        Segments_Features = []
        num = 32
        thirty2_shots = np.round(np.linspace(0, len(features) - 1, num=num+1)).astype(int)
        for ss, ee in zip(thirty2_shots[:-1], thirty2_shots[1:]):
            if ss == ee:
                temp_vect = features[min(ss, features.shape[0] - 1), :]
            else:
                temp_vect = features[ss:ee, :].mean(axis=0)

            temp_vect = temp_vect / np.linalg.norm(temp_vect)
            if np.linalg.norm == 0:
                logging.error("Feature norm is 0")
                exit()
            if len(temp_vect) != 0:
                Segments_Features.append(temp_vect.tolist())

        feat_path = os.path.join(OUTPUT_FEAT_DIR, video_name.replace("/", "-") + '.npy')
        feat_path.replace("/", "\\")

        print('Saving features and probs for video: %s ...'%video_name)
        np.save(feat_path, Segments_Features)
        
        print('%d: %s has been processed...'%(cnt, video_name))
        try: 
            os.system('rmdir /s /q ' + frame_path)  
        except: 
            pass

if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    print('******--------- Extract I3D features ------*******')
    parser.add_argument('-g', '--GPU', type=int, default=0, help='GPU id')
    parser.add_argument('-of', '--OUTPUT_FEAT_DIR', dest='OUTPUT_FEAT_DIR', type=str,
                        default='C:/deep/I3D-Feature-Extractor/features',
                        help='Output feature path')
    parser.add_argument('-vpf', '--VIDEO_PATH_FILE', type=str,
                        default='Test_Annotation.txt',
                        help='input video list')
    parser.add_argument('-vd', '--VIDEO_DIR', type=str,
                        default='C:/deep/Pytorch_C3D_Feature_Extractor/videos',
                        help='frame directory')

    args = parser.parse_args()
    params = vars(args) # convert to ordinary dict

    OUTPUT_FEAT_DIR = params['OUTPUT_FEAT_DIR']
    VIDEO_PATH_FILE = params['VIDEO_PATH_FILE']
    VIDEO_DIR = params['VIDEO_DIR']
    RUN_GPU = params['GPU']

    resize_w = 224
    resize_h = 224
    L = 64
    batch_size = 1

    # set gpu id
    os.environ['CUDA_VISIBLE_DEVICES'] = str(RUN_GPU)

    feature_extractor()


