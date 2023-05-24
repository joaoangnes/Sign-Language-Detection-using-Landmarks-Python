import os
import cv2
import HandTrackingModule as hm
import pickle

# Use yours directory
DATA_DIR = '/Users/joaoangnes/Documents/Faculdade/Sistema Inteligentes/Sign-Language-Detection-using-Landmarks-Python/images'
if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)

number_of_classes = 20
dataset_size = 300
largura_janela = 640
altura_janela = 480

cap = cv2.VideoCapture(1)
cap.set(3, largura_janela)
cap.set(4, altura_janela)
detector = hm.handDetector(static_image_mode=False, max_num_hands=1)

for j in range(number_of_classes):
    if not os.path.exists(os.path.join(DATA_DIR, str(j))):
        os.makedirs(os.path.join(DATA_DIR, str(j)))

    print('Collecting data for class {}'.format(j))

    while True:
        success, img = cap.read()
        cv2.putText(img, 'Ready? Press "S" ! :)', (100, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 255, 0), 3,
                    cv2.LINE_AA)
        cv2.imshow("Image", img)
        
        if cv2.waitKey(1) & 0xFF == ord('s'):
            break
        
    counter = 0
    while counter < dataset_size:
        if counter == 150:
            while True:
                ret, img = cap.read()
                img = detector.findHands(img, draw=False)   
                cv2.putText(img, 'Left Hand Now! Press "S" ! :)', (100, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 255, 0), 3,
                            cv2.LINE_AA)
                cv2.imshow('Image', img)
                cv2.imwrite(os.path.join(DATA_DIR, str(j), '{}.jpg'.format(counter)), img)
                if cv2.waitKey(25) == ord('s'):
                    break
                
        ret, img = cap.read()
        img = detector.findHands(img, draw=False)       
        cv2.imshow('Image', img)
        cv2.imwrite(os.path.join(DATA_DIR, str(j), '{}.jpg'.format(counter)), img)
        counter += 1
        cv2.waitKey(25)
    
cap.release()
cv2.destroyAllWindows() 
