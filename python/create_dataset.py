import os
import pickle
import mediapipe as mp
import cv2

DATA_DIR = './data'



mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

body = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)

data = []
labels = []

for class_dir in os.listdir(DATA_DIR):
    class_path = os.path.join(DATA_DIR, class_dir)
    if os.path.isdir(class_path):
        for img_path in os.listdir(class_path):
            data_aux = []
            img = cv2.imread(os.path.join(class_path, img_path))
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            results = body.process(img_rgb)
            
            if results.pose_landmarks:
                for body_landmark in results.pose_landmarks.landmark:
                    x = body_landmark.x
                    y = body_landmark.y
                    z = body_landmark.z

                    data_aux.append(x)
                    data_aux.append(y)

            data.append(data_aux)
            labels.append(class_dir)

f = open('data.pickle', 'wb')
pickle.dump({'data': data, 'labels': labels}, f)
f.close()
