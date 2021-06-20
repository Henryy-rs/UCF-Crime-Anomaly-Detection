# coding: utf-8
# from data_provider import *
from C3D_model import *
import torchvision
import torch
from torch.autograd import Variable
import torch.nn.functional as F
import argparse
import os 
from torch import save, load
import pickle
import time
import numpy as np
import PIL.Image as Image
import skimage.io as io
import h5py
from PIL import Image
from get_frame_gpu import RgbByFrame


def feature_extractor():
	net = C3D(487)
	print('net', net)
	## Loading pretrained model from sports and finetune the last layer
	net.load_state_dict(torch.load('C:/deep/Pytorch_C3D_Feature_Extractor/data/pretrained_models/c3d.pickle'))
	if RUN_GPU : 
		net.cuda(0)
		net.eval()
		print('net', net)
	feature_dim = 4096 if EXTRACTED_LAYER != 5 else 8192

	# read video list from the txt list
	video_list_file = args.video_list_file
	video_list = open(video_list_file).readlines()
	video_list = [item.strip() for item in video_list]

	gpu_id = args.gpu_id

	if not os.path.isdir(OUTPUT_DIR):
		os.mkdir(OUTPUT_DIR)
	f = h5py.File(os.path.join(OUTPUT_DIR, OUTPUT_NAME), 'a')

	frame = RgbByFrame()

	error_fid = open('error.txt', 'w')
	for video_name in video_list: 
		video_name = video_name.split(" ")[0]	#added
		video_path = os.path.join(VIDEO_DIR, video_name)
		print('video_path', video_path)
		frame.set_video(video_path)
		print('Extracting features ...')
		total_frames = len(frame)
		valid_frames = total_frames / nb_frames * nb_frames
		n_feat = valid_frames / nb_frames
		n_batch = n_feat / BATCH_SIZE 
		if n_feat - n_batch*BATCH_SIZE > 0:
			n_batch = n_batch + 1
		n_batch = int(n_batch)
		n_feat = int(n_feat)
		features = []
		for i in range(n_batch-1):
			input_blobs = []
			for j in range(BATCH_SIZE):
				clip = []
				clip = np.array([frame.get_rgb_frame() for k in range((i*BATCH_SIZE+j) * nb_frames+1, min((i*BATCH_SIZE+j+1) * nb_frames+1, valid_frames+1))])
				input_blobs.append(clip)
			input_blobs = np.array(input_blobs, dtype='float32')
			print('input_blobs_shape', input_blobs.shape)
			input_blobs = torch.from_numpy(np.float32(input_blobs.transpose(0, 4, 1, 2, 3)))
			input_blobs = Variable(input_blobs).cuda() if RUN_GPU else Variable(input_blobs)
			_, batch_output = net(input_blobs, EXTRACTED_LAYER)	
			batch_feature  = (batch_output.data).cpu()
			features.append(batch_feature) 
			del batch_feature
			torch.cuda.empty_cache()
		
		# The last batch
		input_blobs = []
		for j in range(n_feat-(n_batch-1)*BATCH_SIZE):
			clip = []
			clip = np.array([frame.get_rgb_frame() for k in range((i*BATCH_SIZE+j) * nb_frames+1, min((i*BATCH_SIZE+j+1) * nb_frames+1, valid_frames+1))])
			input_blobs.append(clip)
		input_blobs = np.array(input_blobs, dtype='float32')
		print('input_blobs_shape', input_blobs.shape)
		input_blobs = torch.from_numpy(np.float32(input_blobs.transpose(0, 4, 1, 2, 3)))
		input_blobs = Variable(input_blobs).cuda() if RUN_GPU else Variable(input_blobs)
		_, batch_output = net(input_blobs, EXTRACTED_LAYER)
		batch_feature  = (batch_output.data).cpu()
		features.append(batch_feature)
		features = torch.cat(features, 0)
		features = features.numpy()
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

		fgroup = f.create_group(video_name)
		fgroup.create_dataset('c3d_features', data=Segments_Features)
		fgroup.create_dataset('total_frames', data=np.array(total_frames))
		fgroup.create_dataset('valid_frames', data=np.array(valid_frames))

		print('%s has been processed...'%video_name)

	f.close()

if __name__ == "__main__":

	parser = argparse.ArgumentParser()

	print ('******--------- Extract C3D features ------*******')
	parser.add_argument('-o', '--OUTPUT_DIR', dest='OUTPUT_DIR', type=str, default='./output_frm/', help='Output file name')
	parser.add_argument('-l', '--EXTRACTED_LAYER', dest='EXTRACTED_LAYER', type=int, choices=[5, 6, 7], default=6, help='Feature extractor layer')
	parser.add_argument('-i', '--VIDEO_DIR', dest='VIDEO_DIR', type = str, help='Input Video directory')
	parser.add_argument('-gpu', '--gpu', dest='GPU', action = 'store_true', help='Run GPU?')
	parser.add_argument('--OUTPUT_NAME', default='c3d_features.hdf5', help='The output name of the hdf5 features')
	parser.add_argument('-b', '--BATCH_SIZE', default=30, help='the batch size')
	parser.add_argument('-id', '--gpu_id', default=0, type=int)
	parser.add_argument('-p', '--video_list_file', type=str, help='the video name list')

	args = parser.parse_args()
	params = vars(args) # convert to ordinary dict
	print('parsed parameters:')

	OUTPUT_DIR = params['OUTPUT_DIR']
	EXTRACTED_LAYER = params['EXTRACTED_LAYER']
	VIDEO_DIR = params['VIDEO_DIR']
	RUN_GPU = params['GPU']
	OUTPUT_NAME = params['OUTPUT_NAME']
	BATCH_SIZE = params['BATCH_SIZE']
	BATCH_SIZE = int(BATCH_SIZE) 	#added
	feature_extractor()


