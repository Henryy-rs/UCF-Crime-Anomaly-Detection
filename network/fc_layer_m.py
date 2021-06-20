import torch
from torch import nn

import numpy as np


class AnomalyDetector(nn.Module):
    def __init__(self, input_dim=4096):
        super(AnomalyDetector, self).__init__()
        self.fc1 = nn.Linear(input_dim, 512)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(0.6)

        self.fc2 = nn.Linear(512, 32)
        self.dropout2 = nn.Dropout(0.6)

        self.fc3 = nn.Linear(32, 1)
        self.sig = nn.Sigmoid()

        # In the original keras code they use "glorot_normal"
        # As I understand, this is the same as xavier normal in Pytorch
        nn.init.xavier_normal_(self.fc1.weight)
        nn.init.xavier_normal_(self.fc2.weight)
        nn.init.xavier_normal_(self.fc3.weight)

    def forward(self, x):
        x = self.dropout1(self.relu1(self.fc1(x)))
        x = self.dropout2(self.fc2(x))
        x = self.sig(self.fc3(x))
        return x


def custom_objective(y_pred, y_true):
    # y_pred (batch_size, 32, 1)
    # y_true (batch_size)
    lambdas = 8e-5

    normal_vids_indices = torch.where(y_true == 0)
    anomal_vids_indices = torch.where(y_true == 1)

    

    normal_segments_scores = y_pred[normal_vids_indices]  # (batch/2, 32, 1)
    anomal_segments_scores = y_pred[anomal_vids_indices]  # (batch/2, 32, 1)
    #########################################################################
    # add opticalflow movement

    video_list = open("training_dataset.txt").readlines()
    video_list = [item.strip() for item in video_list]
    normal_vids_indices = normal_vids_indices[0].cpu().numpy()
    anomal_vids_indices = anomal_vids_indices[0].cpu().numpy()

    k = 0.5
    if torch.cuda.is_available():  
        dev = "cuda:0" 
    else:  
        dev = "cpu"  
    device = torch.device(dev)

    normal_flow_term = torch.zeros([30, 32])
    anomal_flow_term = torch.zeros([30, 32])
    print("cuda?",str(normal_flow_term.is_cuda))
    for i in normal_vids_indices:
        optical_flow_values =  np.load("./optical_features/"+video_list[i].replace('/', '-') + ".npy")
        #print("optical: " , optical_flow_values)
        i_nor = int(i/2)
        for index, value in enumerate(optical_flow_values):
            #if index > 31:
            #    break
            # 첫 segment이면, 이 경우 변화량을 계산할 수 없다
            if index == 0:
                normal_flow_term[i_nor][index] = 0
            else:
                #print(optical_flow_values[index], optical_flow_values[index])
                if optical_flow_values[index-1] != 0:
                    variation = abs(round(optical_flow_values[index]/optical_flow_values[index-1] - 1, 4))
                else:
                    variation = 0
                #print("variation: ", variation)
                if variation >= 10:
                    normal_flow_term[i_nor][index] = 0
                else:
                    normal_flow_term[i_nor][index] = variation

    for i in anomal_vids_indices:
        optical_flow_values =  np.load("./optical_features/"+video_list[i].replace('/', '-') + ".npy")
        #print("optical: " , optical_flow_values)
        i_ano = int(i/2)
        for index, value in enumerate(optical_flow_values):
            #if index > 31:
            #    break
            if index == 0:
                anomal_flow_term[i_ano][index] = 0
            else:
                if optical_flow_values[index-1] != 0:
                    variation = abs(round(optical_flow_values[index]/optical_flow_values[index-1]-1, 4))
                else:
                    variation = 0
                if variation >= 10:
                    anomal_flow_term[i_ano][index] = 0
                else:
                    anomal_flow_term[i_ano][index] = variation

    normal_flow_term = k * normal_flow_term
    anomal_flow_term = k * anomal_flow_term

    #print(normal_flow_term)
    normal_flow_term = normal_flow_term.to(device)
    anomal_flow_term = anomal_flow_term.to(device)
    #########################################################################


    # just for reducing the last dimension
    normal_segments_scores = torch.sum(normal_segments_scores, dim=(-1,))  # (batch/2, 32)
    anomal_segments_scores = torch.sum(anomal_segments_scores, dim=(-1,))  # (batch/2, 32)

    # add optical flow term
    normal_segments_scores = torch.add(normal_segments_scores, normal_flow_term)
    anomal_segments_scores = torch.add(anomal_segments_scores, anomal_flow_term)

    # get the max score for each video
    normal_segments_scores_maxes = normal_segments_scores.max(dim=-1)[0]
    anomal_segments_scores_maxes = anomal_segments_scores.max(dim=-1)[0]

    hinge_loss = 1 - anomal_segments_scores_maxes + normal_segments_scores_maxes
    hinge_loss = torch.max(hinge_loss, torch.zeros_like(hinge_loss))

    """
    Smoothness of anomalous video
    """
    smoothed_scores = anomal_segments_scores[:, 1:] - anomal_segments_scores[:, :-1]
    smoothed_scores_squared = smoothed_scores.pow(2)
    smoothness_loss = smoothed_scores_squared.sum(dim=-1)

    """
    Sparsity of anomalous video
    """
    sparsity_loss = anomal_segments_scores.sum(dim=-1)

    final_loss = (hinge_loss + lambdas*smoothness_loss + lambdas*sparsity_loss).mean()
    return final_loss


class RegularizedLoss(torch.nn.Module):
    def __init__(self, model, original_objective, lambdas=0.001):
        super(RegularizedLoss, self).__init__()
        self.lambdas = lambdas
        self.model = model
        self.objective = original_objective

    def forward(self, y_pred, y_true):
        # loss
        # Our loss is defined with respect to l2 regularization, as used in the original keras code
        fc1_params = torch.cat(tuple([x.view(-1) for x in self.model.fc1.parameters()]))
        fc2_params = torch.cat(tuple([x.view(-1) for x in self.model.fc2.parameters()]))
        fc3_params = torch.cat(tuple([x.view(-1) for x in self.model.fc3.parameters()]))

        l1_regularization = self.lambdas * torch.norm(fc1_params, p=2)
        l2_regularization = self.lambdas * torch.norm(fc2_params, p=2)
        l3_regularization = self.lambdas * torch.norm(fc3_params, p=2)

        return self.objective(y_pred, y_true) + l1_regularization + l2_regularization + l3_regularization

