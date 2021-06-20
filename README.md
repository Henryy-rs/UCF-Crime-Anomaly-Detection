# UCF-Crime-Anomaly-Detection
CCTV 영상 이상 탐지 모델 개발 및 성능 비교 분석

2021년 1학기 데이터분석캡스톤디자인



> - Library:
> TensorFlow, PyTorch, OpenCV, h5py
> - Environment:
> Windows10, Python3.8, CUDA 10.2/9.2
> - Hardware:
> Intel i7-9750H, GTX 1660Ti
# Overview
  최근 코로나 19 시대를 맞이하여 언택트 트렌드가 확산되면서 다양한 업종의 무인매장이 들어서고 있다. 이에 따라 무인매장에서 도난을 비롯한 사건사고에 관한 소식이 끊이지 않고 있다. 궁극적으로 무인매장 CCTV를 활용하여 사건사고 발생 시 대응할 수 있는 보안솔루션을 개발하는 것이 목표이지만, 이에 앞서 본 과제에서는 CCTV 영상 이상 탐지 모델 몇 가지를 만들어 성능을 측정하고 비교해보았다. 

# Dataset

UCF Crime Dataset
https://www.crcv.ucf.edu/projects/real-world/
![image](https://user-images.githubusercontent.com/28619620/122185184-ddcdd500-cec7-11eb-904f-a7dc2b954def.png)
- Surveilance Videos(CCTV)
- 1900 videos, 13 types of anomaly
- 950 anomal videos/950 normal videos
- weakly labeled data -> 영상의 anomaly와 normal 여부만 알 수 있고 영상의 어느 시점이 anomaly인지에 대한 정보는 없다.


# Model
![image](https://user-images.githubusercontent.com/28619620/122191171-71ee6b00-cecd-11eb-83ec-2ffde455792e.png)

Real-world anomaly detection in surveillance videos에서 제시된 모델이다. 


- 비디오를 32개의 segments로 쪼개고 각각의 segment에 대해 feature extracion을 한다. 미리 학습된 C3D(3D Convolutional Neural Network), I3D(Inflated 3D Convolutional Neural Network)모델을 이용한다.
- 추출된 segment의 feature를 입력하여 anomaly score를 얻는다.
- fully connect layer는 3층으로 되어있있다. (inuput laye(ReLU) -> layer 1(Linear) -> layer 2(Sigmoid)-> output(amomaly score))
- 학습시 Deep MIL Ranking Model을 이용하여 loss를 계산한다.

논문에서 제시된 기본 모델 하나와 변형된 4가지 모델을 추가해서 성능을 비교해 볼 것이다.

<h2>C3D feature extraction(Pretrained Model)</h2>

You can download pretrained model from here, http://imagelab.ing.unimore.it/files/c3d_pytorch/c3d.pickle

![image](https://user-images.githubusercontent.com/28619620/122191171-71ee6b00-cecd-11eb-83ec-2ffde455792e.png)

<h3>1. C3D Model(RGB)</h3>

> frame의 RGB값을 그대로 사용한다.

> 아래는 feature를 얻기 위한 command이다.

```python c3d_feature_extractor.py -l 6 -i "input videos path" -o "output features path" -gpu -id 0 -p "annotation path" --OUTPUT_NAME "output features name" -b "batch-size"```

<h3>2. C3D Model(Denseflow)</h3>

> - RGB대신 Farneback optical flow를 이용하여 feature extraction한다.
> - (r, g, b) -> (x, y, 0) (which is input for C3D extractor)

```python c3d_denseflow_feature_extractor.py -l 6 -i "input videos path" -o "output features path" -gpu -id 0 -p "All_Annotation.txt" --OUTPUT_NAME "output features name" -b "batch-size"```

<h3>3. C3D Model(RGB + optical movement Value)</h3>

다음과 같은 방법으로 segment안의 점들에서 이동량의 변화를 계산하여 anomaly score에 더해주는 모델이다.
이동량의 변화량이 높을수록 anomaly일 확률이 높다고 판단하여 모델을 만들어보았다.

> - good points들의 Lucas-Kanade optical flow를 계산한다.
> - 이전 frame과 비교하여 good points의 이동량을 계산하고 합친다.
> - 이 합친 값을 frame의 movement value로 활용한다.
> - 학습시 현재 frame과 이전 frame과 비교하여 변화량을 구하고 상수를 곱해서 anomaly score에 더해준다.
> ![image](https://user-images.githubusercontent.com/28619620/122207578-3fe50500-cedd-11eb-84e7-3cde55a413b7.png)

```python get_movement_frame --VIDEO_DIR "video path" --VIDEO_NAME "All_Annotation.txt" --OUTPUT_DIR "output path"```

<h2>I3D feature extraction(Pretrained Model)</h2>

![i3d model](https://user-images.githubusercontent.com/28619620/122194103-18d40680-ced0-11eb-9fac-1086f87c40bb.jpg)

<h3>1. I3D Model(RGB)</h3>

> frame의 RGB값을 그대로 사용한다.

```python i3d_feature_extractor.py -g 0 -of "output features path" --vpf "All_Annotation.txt" -vd "video path"```

```python get_i3d_h5.py -g 0 -OUTPUT_FILE_NAME "output hdf5 file name" --FEATURES_DIR "path where .npy"```

<h3>2. I3D Model(RGB, Add optical movement Value)</h3>

> C3D 모델과 같은 방식으로 Movement를 사용한다.

```python get_movement_frame --VIDEO_DIR "video path" --VIDEO_NAME "All_Annotation.txt" --OUTPUT_DIR "output path"```

# Training

> - Training Dataset: 810 anomaly videos, 800 normal videos
> - 20000 epoch(default)
> - Multiple Instance Learning


weakly label data를 사용하기 때문에 어느 시점이 anomaly인지 알 수 없다. 이런 상황에서 학습을 하기 위해  방법을 사용한다.

>-  Bag Instance안에 있는 32개의 segments의 anomaly score를 구한다.
>- anomaly score가 높은 anomaly video segmet는 anomaly 특성(true positive)을 지니고 있고, anomlay score가 높은 normal video segment는 anomaly와 비슷한 특성을 가지고 있지만, anomaly가 >아니므로 false alarm(false positive)을 생성할 수 있다. 그러므로 각 Bag instance안에 있고 anomaly score가 최대인 segment를 이용하여 loss를 계산한다.
>- Positive Bag와 Negative Bag Instance의 Score를 이용해 Loss를 계산한다. 

<h3>Objective Function</h3>

> MIL Ranking Loss
>
> ![image](https://user-images.githubusercontent.com/28619620/122366514-98c4a400-cf96-11eb-8042-e5a2324e36a4.png)
>
>More details are in the paper.

<h3>Optimization</h3>

>
>adagrad, learning rate = 0.001

<h3>Code</h3>

> <b>C3D Model(RGB/Denseflow)</b>
> 
```python TrainingAnomalyDetector_public.py --features_path "features path" --annotation_path "Train_Annotation.txt" --extractor_type "c3d"```

> <b>C3D Model(RGB, add optical movement value)</b>
> 
> You have to change "from network.fc_layer" to "from network.fc_layer_m" in TrainingAnomalyDetector_public.py

```python TrainingAnomalyDetector_public.py --features_path "features path" --annotation_path "Train_Annotation.txt" --extractor_type "c3d"```

> <b>I3D Model(RGB)</b> *Best Performance*

```python TrainingAnomalyDetector_public.py --features_path "features path" --annotation_path "Train_Annotation.txt" --extractor_type "i3d"```

> <b>I3D Model(RGB + optical movement)</b>
>
> You have to change "from network.fc_layer" to "from network.fc_layer_m" in TrainingAnomalyDetector_public.py
 
```python TrainingAnomalyDetector_public.py --features_path "features path" --annotation_path "Train_Annotation.txt" --extractor_type "i3d"```
# Testing

>각 모델의 결과 비교, 논문과 비교
>
![image](https://user-images.githubusercontent.com/28619620/122355279-bab92900-cf8c-11eb-9e71-5d30a6023aee.png)
![image](https://user-images.githubusercontent.com/28619620/122355706-14b9ee80-cf8d-11eb-8f69-7e71a3150e89.png)

<h3>C3D(Denseflow)</h3>

![denseflow](https://user-images.githubusercontent.com/28619620/122676269-5e6a3980-d218-11eb-994a-a68405850fa8.png)

<h3>C3D(RGB)</h3>

![roc_auc_c3d_40k](https://user-images.githubusercontent.com/28619620/122676280-69bd6500-d218-11eb-88b6-29750123c277.png)


<h3>C3D(RGB + optical flow)</h3>

![roc_auc_c3d_40k_movement](https://user-images.githubusercontent.com/28619620/122676286-704bdc80-d218-11eb-935f-1626fe816bb9.png)

<h3>I3D(RGB)</h3>

![roc_auc_i3d_40k (2)](https://user-images.githubusercontent.com/28619620/122676294-7cd03500-d218-11eb-9d3c-c8c4c80378d7.png)

<h3>I3D(RGB + optical flow)</h3>

![image](https://user-images.githubusercontent.com/28619620/122676522-6d9db700-d219-11eb-8f52-4baa54a5ac25.png)


![image](https://user-images.githubusercontent.com/28619620/122676331-a9844c80-d218-11eb-8e29-e3be70c002ca.png)

>1. C3D 모델의 성능이 논문보다 소폭 낮은 이유

(1)320X240 영상을 224X224로 resize하여 특징추출하였다.
논문에서는 320X240 영상을 사용했다는 언급만 있다.

(2)특징추출에 사용된 pretrained model가 다를 수 있다.
(Sports-1M)

(3)PyTorch로 구현된 코드를 수정하여 테스트했다.
논문에서는 theano, keras를 이용해 코드를 작성했다.

>2. Optical flow 모델의 성능차이가 나지 않는 이유

(1) Noise, Lucas-Kanade Optical flow의 한계

(2)C3D feature에 이미 이동량이 반영되어 있을 수 있다. 
C3D는 segment의 temporal information을 반영할 수 있다.

(3)이동량이 모든 도메인에서 적용되진 않는다.

# Conclusion

 Lucas-Kanade Optical flow 적용의 효과는 없었다. I3D feature extraction 기법을 사용했을 때 가장 좋은 성능을 얻을 수 있었으며 이는 논문에 제시된 C3D(RGB)모델보다 좋은 성능이다. 
 
# What's Next

Realtime Detector

# Reference

https://github.com/yyuanad/Pytorch_C3D_Feature_Extractor

https://github.com/ekosman/AnomalyDetectionCVPR2018-Pytorch

https://github.com/JaywongWang/I3D-Feature-Extractor

 W. Sultani, C. Chen, and M. Shah. “Real-world anomaly detection in surveillance videos”. 
arXiv preprint arXiv:1801.04264, 2018
