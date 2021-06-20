import numpy as np
import cv2
import math
import argparse
import os

#print(cv2.__version__)

def get_32_opt():
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
    
    cap = cv2.VideoCapture(os.path.join(VIDEO_DIR, VIDEO_NAME))
    length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    #frame_per_seg = int(length/32)
    frame_per_seg = 16
    #print("frame_per_seg: ", str(frame_per_seg))
    #print("frame_length: ", str(length))

    # params for ShiTomasi corner detection
    feature_params = dict( maxCorners = 100,
                        qualityLevel = 0.3,
                        minDistance = 7,
                        blockSize = 7 )

    # Parameters for lucas kanade optical flow
    lk_params = dict( winSize  = (15,15),
                    maxLevel = 2,
                    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

    # Create some random colors
    #color = np.random.randint(0,255,(100,3))

    # Take first frame and find corners in it
    ret, old_frame = cap.read()
    old_gray = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)
    p0 = cv2.goodFeaturesToTrack(old_gray, mask = None, **feature_params)
    #print(p0)

    # Create a mask image for drawing purposes
    #mask = np.zeros_like(old_frame)
    L2_distance_seg = 0
    L2_distance_list = []
    j = 0
    while(1):
        ret, frame = cap.read()
        L2_distance = 0
        if frame is None:
            break
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # calculate optical flow
        if p0 is None:
            p0 = cv2.goodFeaturesToTrack(frame_gray, mask = None, **feature_params)
            j += 1
            if j >= frame_per_seg:
                j = 0
                L2_distance_list.append(L2_distance_seg)
                L2_distance_seg = 0
            continue
        p1, st, err = cv2.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, **lk_params)
        
        if p1 is None:
            p0 = cv2.goodFeaturesToTrack(old_gray, mask=None, **feature_params)
            if p0 is None:
                p0 = cv2.goodFeaturesToTrack(frame_gray, mask = None, **feature_params)
                j += 1
                if j >= frame_per_seg:
                    j = 0
                    L2_distance_list.append(L2_distance_seg)
                    L2_distance_seg = 0
                continue
            p1, st, err = cv2.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, **lk_params)
        # Select good points
        good_new = p1[st==1]
        good_old = p0[st==1]
        
        for i,(new,old) in enumerate(zip(good_new,good_old)):
            a,b = new.ravel()
            c,d = old.ravel()
            L2_distance += math.sqrt((a-c)**2+(b-d)**2)
        
        L2_distance_seg += L2_distance
        j += 1
        if j >= frame_per_seg:
            j = 0
            #print(L2_distance_seg)
            L2_distance_list.append(L2_distance_seg)
            L2_distance_seg = 0

        # Now update the previous frame and previous points
        old_gray = frame_gray.copy()
        p0 = good_new.reshape(-1,1,2)

    #make 32 features
    num=32
    flow_features = []
    L2_distance_list = np.array(L2_distance_list)
    thirty2_shots = np.round(np.linspace(0, len(L2_distance_list) - 1, num=num+1)).astype(int)
    for ss, ee in zip(thirty2_shots[:-1], thirty2_shots[1:]):
	    if ss == ee:
		    temp_vect = L2_distance_list[min(ss, L2_distance_list.shape[0] - 1)]
	    else:
		    temp_vect = L2_distance_list[ss:ee].mean(axis=0)

	    #temp_vect = temp_vect / np.linalg.norm(temp_vect)
	    if np.linalg.norm == 0:
				#logging.error("Feature norm is 0")
		    exit()
	    #if len(temp_vect) != 0:
	    flow_features.append(round(temp_vect , 4))

    feature_path = os.path.join(OUTPUT_DIR, VIDEO_NAME.replace('/', '-') + '.npy')
    
    print(VIDEO_NAME)

    assert len(flow_features) == 32, "ERROR:features length is not 32"

    np.save(feature_path, flow_features)
    cap.release()
    

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--VIDEO_DIR', dest='VIDEO_DIR', type=str, default='./temp')
    parser.add_argument('--VIDEO_NAME', dest='VIDEO_NAME', type=str, default='./temp')
    parser.add_argument('--OUTPUT_DIR', dest='OUTPUT_DIR', type=str, default='./temp')
    args = parser.parse_args()
    params = vars(args)

    VIDEO_DIR = params['VIDEO_DIR']
    VIDEO_NAME = params['VIDEO_NAME']
    OUTPUT_DIR = params['OUTPUT_DIR']
    
    get_32_opt()