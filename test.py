import numpy as np
import cv2
import time
import datetime
from matplotlib import pyplot as plt


cap = cv2.VideoCapture('./UCSDped_video/ped1/training_videos/02.avi')
# cap = cv2.VideoCapture('./testnegatice.avi')
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3))
fgbg = cv2.createBackgroundSubtractorMOG2()
fourcc = cv2.VideoWriter_fourcc(*'XVID')
frame1 = np.zeros((158, 238))
# out = cv2.VideoWriter(datetime.datetime.now().strftime("%A_%d_%B_%Y_%I_%M_%S%p")+'.avi', fourcc, 5.0, np.shape(frame1))


while(1):
    ret, frame = cap.read()
    if not ret:
        break
    cv2.bilateralFilter(frame, 0, 100, 10)
    fgmask = fgbg.apply(frame)
    # fgmask_blur = cv2.GaussianBlur(fgmask, (13, 13), 0, sigmaY=0)
    # fgmask_blur = cv2.medianBlur(fgmask, ksize=3)
    fgmask_blur = cv2.bilateralFilter(fgmask, 0, 100, 10)
    cv2.imshow('mog', fgmask)
    cv2.imshow('blur', fgmask_blur)
    (_, cnts, _) = cv2.findContours(fgmask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    maxArea = 0
    for c in cnts:
        if cv2.contourArea(c) > 15 and cv2.contourArea(c) < 1000:
            m=c
            (x, y, w, h) = cv2.boundingRect(c)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            # out.write(frame)
            cv2.imshow('frame',frame)
    k = cv2.waitKey(30)&0xff
    if k==27:
        break
# out.release()
cap.release()
cv2.destroyAllWindows()

"""
with open('/Users/chenchengwei/Desktop/retina_ganomaly_abnormal/frame_90') as f:
    scores=[]
    for each in f.readlines():
        scores.append(float(each.strip().split(',')[0]))

plt.plot(scores)
plt.show()
"""


"""
fig = plt.figure()
ax1 = fig.add_subplot(3,2,1)
ax2 = fig.add_subplot(3,2,2)
ax3 = fig.add_subplot(3,2,3)
ax4 = fig.add_subplot(3,2,4)
ax5 = fig.add_subplot(3,2,5)
ax6 = fig.add_subplot(3,2,6)


count=0
with open('/Users/chenchengwei/Desktop/retina_ganomaly_abnormal/video36_enlarged') as f:
    for num,each in enumerate(f.readlines()[100:120]):
        scores = []
        for each_num in each.split(','):
            scores.append(float(each_num))
        ax1 = fig.add_subplot(3, 2, num+1)
        ax1.plot(scores)
        count = count + 1
        if count==6:
            break



plt.show()
"""