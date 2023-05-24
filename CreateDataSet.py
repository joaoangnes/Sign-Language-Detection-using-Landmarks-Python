import os
import pickle
import cv2
import HandTrackingModule as hm

detector = hm.handDetector(static_image_mode=True, max_num_hands=1)

DATA_DIR = '/Users/joaoangnes/Documents/Faculdade/Sistema Inteligentes/Sign-Language-Detection-using-Landmarks-Python/images'

data = []
classes = []

for dir_ in os.listdir(DATA_DIR):
    for img_path in os.listdir(os.path.join(DATA_DIR, dir_)):
        img = cv2.imread(os.path.join(DATA_DIR, dir_, img_path))
        lm_list, img = detector.findHands(img, draw=False, getPosition=True) 
        if len(lm_list) != 0:
            data.append(lm_list)
            classes.append(dir_)

f = open('/Users/joaoangnes/Documents/Faculdade/Sistema Inteligentes/Sign-Language-Detection-using-Landmarks-Python/data.pickle', 'wb')
pickle.dump({'data': data, 'class': classes}, f)
f.close()