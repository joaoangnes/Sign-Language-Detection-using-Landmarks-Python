import cv2
import mediapipe as mp
import time
import pandas as pd
import numpy as np
import math


class handDetector():
    def __init__(self, static_image_mode=False, max_num_hands=2, model_complexity = 0, min_detection_confidence=0.7, min_tracking_confidence=0.7):
        self.static_image_mode = static_image_mode
        self.max_num_hands = max_num_hands
        self.model_complexity = model_complexity
        self.min_detection_confidence = min_detection_confidence
        self.min_tracking_confidence = min_tracking_confidence
        
        self.mpHands = mp.solutions.hands
        self.hands   = self.mpHands.Hands(static_image_mode = self.static_image_mode, 
                                          max_num_hands = self.max_num_hands, 
                                          model_complexity = self.model_complexity,
                                          min_detection_confidence = self.min_detection_confidence, 
                                          min_tracking_confidence = self.min_tracking_confidence) # Default parameters
        self.mpDraw  = mp.solutions.drawing_utils # Solução para desenhar os pontos da mão

    def findHands(self, img, draw=True, flipType=True, getPosition = False):
        """
        Finds hands in a BGR image.
        :param img: Image to find the hands in.
        :param draw: Flag to draw the output on the image.
        :return: Image with or without drawings
        """
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(imgRGB)
        # allHands = []
        height, width, c = img.shape
        lm_list = []
        if self.results.multi_hand_landmarks:
            for handType, handLms in zip(self.results.multi_handedness, self.results.multi_hand_landmarks):
                # myHand = {}
                xList = []
                yList = []
                for id, lm in enumerate(handLms.landmark):
                    px, py = int(lm.x * width), int(lm.y * height)
                    lm_list.append([px, py])
                    xList.append(px)
                    yList.append(py)

                ## bbox
                xmin, xmax = min(xList), max(xList)
                ymin, ymax = min(yList), max(yList)
                boxW, boxH = xmax - xmin, ymax - ymin
                bbox = xmin, ymin, boxW, boxH
                cx, cy = bbox[0] + (bbox[2] // 2), \
                         bbox[1] + (bbox[3] // 2)

                # myHand["lmList"] = lm_list
                # myHand["bbox"] = bbox
                # myHand["center"] = (cx, cy)

                if flipType:
                    if handType.classification[0].label == "Right":
                       type = "Left"
                    else:
                       type = "Right"
                else:
                   type = handType.classification[0].label
                # allHands.append(myHand)

                ## draw
                if draw:
                    self.mpDraw.draw_landmarks(img, handLms, self.mpHands.HAND_CONNECTIONS)
                    cv2.rectangle(img, (bbox[0] - 20, bbox[1] - 20),
                                  (bbox[0] + bbox[2] + 20, bbox[1] + bbox[3] + 20),
                                  (116, 185, 255), 2)
                    cv2.putText(img,type, (bbox[0] - 30, bbox[1] - 30), cv2.FONT_HERSHEY_PLAIN,
                                2, (116, 185, 255), 2)
        if getPosition:
            return lm_list, img
        else:
            return img
                       
    def getColumnsData(self):
        # Lista com os nomes das novas colunas
        new_cols = []
        for i in range(21):
            # new_cols.append(f'ponto {i}')
            new_cols.append(f'x{i}')
            new_cols.append(f'y{i}')
        
        return new_cols
    
    def transformData(self, data):
        df_data = pd.DataFrame()
        
        # Transformando cada dados dos arrays em colunas do dataframe
        for lista in data:
            linha_list = np.asarray(lista).reshape(-1)
            linha = pd.DataFrame([linha_list])
            df_data = pd.concat([df_data, linha], ignore_index=True)         

        col = self.getColumnsData()
        # Renomeia as colunas do dataframe com as novas colunas coletadas
        df_data.columns = col
        
        return df_data
    
    def normalizeData(self, data):
        # ========= NORMALIZCÃO DOS DADOS ========================
        from pickle import load
        normalization_model = load(open('/Users/joaoangnes/Documents/Faculdade/Sistema Inteligentes/Sign-Language-Detection-using-Landmarks-Python/normalization.model','rb'))

        dt_num_normalized = normalization_model.transform(data)
        df_data = pd.DataFrame(dt_num_normalized, columns=data.columns)
        
        return df_data
        
    
def main():
    largura_janela = 640
    altura_janela = 480
    cap = cv2.VideoCapture(1)
    cap.set(3, largura_janela)
    cap.set(4, altura_janela)
    detector = handDetector(static_image_mode=False, max_num_hands=1)

    while True:
        success, img = cap.read()
        img = detector.findHands(img, draw=True)
        cv2.imshow("Image", img)
        key = cv2.waitKey(1)
        
if __name__ == "__main__":
    main()