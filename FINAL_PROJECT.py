import cv2 as cv
import time
import numpy as np
import HandModule as hm
import math
from ctypes import cast,POINTER
from comtypes import CLSCTX_ALL 
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume

import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import absl.logging
absl.logging.set_verbosity(absl.logging.INFO)

import tensorflow as tf
tf.get_logger().setLevel('ERROR')


wCam,hCam=640,480

video_cap=cv.VideoCapture(0)
video_cap.set(3,wCam)
video_cap.set(4,hCam)

cTime = 0
pTime = 0

detect=hm.handDetector()

devices=AudioUtilities.GetSpeakers()
interface=devices.Activate(IAudioEndpointVolume._iid_,CLSCTX_ALL,None)
volume=cast(interface,POINTER(IAudioEndpointVolume))
volRange=volume.GetVolumeRange()
minVol=volRange[0]
maxVol=volRange[1]
volBar=300
vol=0



while True:
    ret ,video_data=video_cap.read()
    video_data=detect.findHands(video_data)
    lmList=detect.findPosition(video_data,draw=False)
    if len(lmList) !=0:
        #print(lmList[4],lmList[8])

        x1,y1=lmList[4][1],lmList[4][2]
        x2,y2=lmList[8][1],lmList[8][2]
        cx,cy=(x1+x2)//2,(y1+y2)//2

        cv.circle(video_data,(x1,y1),15,(255,0,255),cv.FILLED)
        cv.circle(video_data,(x2,y2),15,(255,0,255),cv.FILLED)
        cv.line(video_data,(x1,y1),(x2,y2),(255,0,255),3)

        length=math.hypot(x2-x1,y2-y1)
        #print(length)

        vol=np.interp(length,[25,200],[minVol,maxVol])
        volBar=np.interp(length,[25,200],[400,150])
        #print(vol)
        volume.SetMasterVolumeLevel(vol,None)

        if length<=50:
            cv.circle(video_data,(cx,cy),15,(0,255,0),cv.FILLED)

    cv.rectangle(video_data,(50,150),(85,400),(0,255,0),3)
    cv.rectangle(video_data,(50,int(volBar)),(85,400),(0,255,0),cv.FILLED)



    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime

    cv.putText(video_data,f'FPS: {int(fps)}', (50,70), cv.FONT_HERSHEY_PLAIN, 3, (255,0,0), 4)
    cv.imshow('video',video_data)
    if cv.waitKey(1) == ord("a"):
        break