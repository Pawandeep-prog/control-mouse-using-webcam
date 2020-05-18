import matplotlib.pyplot as plt 
from pynput.mouse import Button, Controller
from tensorflow.keras.models import load_model
import cv2
import numpy as np
import time

control = Controller()

x1, y1 = -1, -1
k = -1
flag = True

cap = cv2.VideoCapture(0)
model = load_model('handmodel_fingers_model.hdf')
res_ = ['zero', 'one', 'two', 'three', 'four', 'five']

def draww(event, x, y, flag, param):
    global x1, y1, k
    if event == cv2.EVENT_LBUTTONDOWN:
        x1 = x
        y1 = y
        k = 1

cv2.namedWindow("img")
cv2.setMouseCallback("img", draww)

while True:
    _, selec = cap.read()
    selec = cv2.flip(selec, 1)
    selec = selec[50:400,250:640, :]
    old_gray = cv2.cvtColor(selec, cv2.COLOR_BGR2GRAY)
    cv2.imshow("img", selec)    
    if k == 1 or cv2.waitKey(1) == 27:
        cv2.destroyAllWindows()
        break


old_pts = np.array([[x1, y1]], dtype=np.float32) 
old_pts = old_pts.reshape(-1,1,2)


mask = np.zeros_like(selec)

control.position = (734,396)
   
while True:
    _, new_img = cap.read()
    new_img = cv2.flip(new_img, 1)
    new_img = new_img[50:400,250:640, :]
    new_gray = cv2.cvtColor(new_img, cv2.COLOR_BGR2GRAY)

    new_pts, status, err = cv2.calcOpticalFlowPyrLK(old_gray,
                                                    new_gray, old_pts, None,
                                                    maxLevel=1,
                                                    criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
                                                              16, 1))
    for  i in new_pts:
        x,y = i.ravel()
        cv2.circle(new_img, (int(x),int(y)), 10, (0,255,0), -1)
        mask = cv2.line(mask, (x,y), (x,y), (0,255,255), 5)
        frame = new_img.copy()
        control.position = (x*4,y*3)
    
    #####################   
    re, bw = cv2.threshold(new_gray, 130, 255, cv2.THRESH_BINARY_INV)
    if x != 0 and y != 0:
        gray = bw[int(y-10):int(y+160), int(x-40):int(x+100)]
        gray = gray / 255.0
        cv2.imshow("gray", gray)
        gray = cv2.resize(gray, (128, 128)) 
        gray = gray.reshape(-1,128,128,1)     
        result=model.predict(gray)
        res = np.argmax(result)
        
        if flag:
            if res == 5:
                control.press(Button.right)
                control.release(Button.right)
                flag = False
        elif res == 2:
            control.press(Button.left)
            control.release(Button.left)
            flag = True
            
        '''
        elif res == 3:
            control.press(Button.left)
            control.release(Button.left)
        elif res == 2:
            control.press(Button.left)
            control.release(Button.left)
        '''
            
            
        cv2.putText(frame, res_[res],(125,125), cv2.FONT_HERSHEY_SIMPLEX, 2,(255,0,0),2)
        cv2.rectangle(frame, (int(x-40),int(y-10)), (int(x+100),int(y+160)), (0,0,0), 2)
    #########################
    
    cv2.imshow("a", mask)
    cv2.imshow('c', frame)

    
    old_gray = new_gray.copy()
    old_pts = new_pts.reshape(-1,1,2)    
    
    if cv2.waitKey(1) == 27:
        break
   

cap.release()
cv2.destroyAllWindows()


print(int(y-20)-int(y+140))
print(int(x-40)-int(x+100))
