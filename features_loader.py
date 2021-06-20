import logging
import os
import random
from os import path
import numpy as np
import torch
from torch.utils import data
#from feature_extractor import read_features
from torch.utils.data.dataset import Dataset
import h5py

class FeaturesLoader(data.Dataset):
    def __init__(self,
                 features_path,
                 features_file_name,
                 annotation_path,
                 bucket_size=30,
                 extractor_type="c3d"):

        super(FeaturesLoader, self).__init__()
        self.features_path = features_path
        self.features_file_name = features_file_name

        #self.features_file = h5py.File(features_path+"/c3d_fc6_features.hdf5", "r")
        #print(self.features_file)

        self.bucket_size = bucket_size
        # load video list
        self.state = 'Normal'
        self.features_list_normal, self.features_list_anomaly = FeaturesLoader._get_features_list(
            features_path=self.features_path,
            annotation_path=annotation_path)

        #self.normal_i, self.anomalous_i = 0, 0
        self.total_i = 0
        self.extractor_type = extractor_type
        self.shuffle()
        print("****************************finish initialization***************************")

    def shuffle(self):
        self.features_list_anomaly = np.random.permutation(self.features_list_anomaly)
        self.features_list_normal = np.random.permutation(self.features_list_normal)

    def __len__(self):
        return self.bucket_size * 2

    def __getitem__(self, index):
        succ = False
        while not succ:
            try:
                feature, label = self.get_feature(index)
                succ = True
            except Exception as e:
                index = np.random.choice(range(0, self.__len__()))
                logging.warning("VideoIter:: ERROR!! (Force using another index:\n{})\n{}".format(index, e))

        return feature, label

    def get_existing_features(self):
        res = []
        for dir in os.listdir(self.features_path):
            dir = path.join(self.features_path, dir)
            if path.isdir(dir):
                for file in os.listdir(dir):
                    file_no_ext = file.split('.')[0]
                    res.append(path.join(dir, file_no_ext))
        return res

    def get_feature(self, index):
        #print("yo")
        if self.state == 'Normal':  # Load a normal video
            idx = random.randint(0, len(self.features_list_normal) - 1)
            feature_subpath = self.features_list_normal[idx]
            label = 0

        elif self.state == 'Anomalous':  # Load an anomalous video
            idx = random.randint(0, len(self.features_list_anomaly) - 1)
            feature_subpath = self.features_list_anomaly[idx]
            label = 1

        #features = read_features(f"{feature_subpath}.txt")
        features = self.read_features(f"{feature_subpath}")
        
        self.state = 'Anomalous' if self.state == 'Normal' else 'Normal'

        return features, label

    @staticmethod
    def _get_features_list(features_path, annotation_path):
        #print("_get_features_list")
        assert os.path.exists(features_path)
        features_list_normal = []
        features_list_anomaly = []
        with open(annotation_path, 'r') as f:
            lines = f.read().splitlines(keepends=False)
            for line in lines:
                items = line.split(' ')[0]
                #video_path = items[0].split('.')[0]
                #file = file.replace('/', os.sep)
                #feature_path = os.path.join(features_path, file)
                feature_path = items #added
                #print(feature_path)
                if 'Normal' in feature_path:
                    features_list_normal.append(feature_path)
                else:
                    features_list_anomaly.append(feature_path)

        return features_list_normal, features_list_anomaly

    # added
    def read_features(self, feature_path):
        # 현재 epoch의 training sample이 저장된다. anomaly score에 movement를 더하기 위해 작성하는 텍스트파일이다.
        f = open("training_dataset.txt", 'a')
        if self.total_i >= 60:
            f.close()
            f = open("training_dataset.txt", 'w')
            self.total_i = 0
        if self.extractor_type == "c3d":
            features_file = h5py.File(self.features_path+"/"+self.features_file_name, "r")
            f.write(feature_path+'\n')
            folder = feature_path.split('/')[0]
            file_name = feature_path.split('/')[1]
            #print(folder, file_name)
            #print(features_file[folder][file_name]['c3d_features']) 
            result = torch.from_numpy(np.array(features_file[folder][file_name]['c3d_features'])).float()
            features_file.close()
        elif self.extractor_type == "i3d":
            features_file = h5py.File(self.features_path+"/"+self.features_file_name, "r")
            f.write(feature_path+'\n')
            folder = feature_path.split('/')[0]
            file_name = feature_path.split('/')[1]
            #print(folder, file_name)
            #print(features_file[folder][file_name]['c3d_features']) 
            result = torch.from_numpy(np.array(features_file[folder][file_name]['i3d_rgb_features'])).float()
            features_file.close()
        f.close()
        self.total_i += 1
        return result
       
        
       
class FeaturesLoaderVal(data.Dataset):
    def __init__(self,
                 features_path,
                 features_file_name,
                 annotation_path,
                 extractor_type="c3d"):

        super(FeaturesLoaderVal, self).__init__()
        self.features_path = features_path
        self.features_file_name = features_file_name
        self.extractor_type = extractor_type
        # load video list
        self.state = 'Normal'
        self.features_list = FeaturesLoaderVal._get_features_list(
            features_path=features_path,
            annotation_path=annotation_path)

    def __len__(self):
        return len(self.features_list)

    def __getitem__(self, index):
        succ = False
        while not succ:
            data = self.get_feature(index)
            succ = True
            """
            try:
                data = self.get_feature(index)
                succ = True
            except Exception as e:
                logging.warning("VideoIter:: ERROR!! (Force using another index:\n{})\n{}".format(index, e))
            """

        return data

    def get_feature(self, index):
        feature_subpath, start_end_couples, length = self.features_list[index]
        print(feature_subpath)
        features = self.read_features(f"{feature_subpath}")
        return features, start_end_couples, length

    @staticmethod
    def _get_features_list(features_path, annotation_path):
        assert os.path.exists(features_path)
        features_list = []
        with open(annotation_path, 'r') as f:
            lines = f.read().splitlines(keepends=False)
            for line in lines:
                start_end_couples = []
                items = line.split()
                anomalies_frames = [int(x) for x in items[3:]]
                start_end_couples.append([anomalies_frames[0], anomalies_frames[1]])
                start_end_couples.append([anomalies_frames[2], anomalies_frames[3]])
                start_end_couples = torch.from_numpy(np.array(start_end_couples))
                #file = items[0].split('.')[0]
                #file = file.replace('/', os.sep)
                #feature_path = os.path.join(features_path, file)
                feature_path = items[0]
                length = int(items[1])
                features_list.append((feature_path, start_end_couples, length))

        return features_list

    def read_features(self, feature_path):
        if self.extractor_type == "c3d":
            features_file = h5py.File(self.features_path+"/"+self.features_file_name, "r")
            folder = feature_path.split('/')[0]
            file_name = feature_path.split('/')[1]
            #print(folder, file_name)
            #print(features_file[folder][file_name]['c3d_features']) 
            result = torch.from_numpy(np.array(features_file[folder][file_name]['c3d_features'])).float()
            features_file.close()
        elif self.extractor_type == "i3d":
            features_file = h5py.File(self.features_path+"/"+self.features_file_name, "r")
            folder = feature_path.split('/')[0]
            file_name = feature_path.split('/')[1]
            #print(folder, file_name)
            #print(features_file[folder][file_name]['c3d_features']) 
            result = torch.from_numpy(np.array(features_file[folder][file_name]['i3d_rgb_features'])).float()
            features_file.close()
        return result


class FeaturesDatasetWrapper(Dataset):
    def __init__(self, features_path,
                 annotation_path,
                 bucket_size=30):
        self.dataset = FeaturesLoader(features_path,
                                      annotation_path,
                                      bucket_size)

    def __getitem__(self, index):
        item = self.dataset[index]
        return {'input': item[0], 'target': item[1]}

    def __len__(self):
        return len(self.dataset)
