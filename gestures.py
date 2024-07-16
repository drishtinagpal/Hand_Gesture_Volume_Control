import cv2 as cv
import mediapipe as mp
import time

video_cap=cv.VideoCapture(0)

cTime=0
pTime=0

mpHands=mp.solutions.hands
hands=mpHands.Hands()
mpDraw=mp.solutions.drawing_utils

while True:
    ret ,video_data=video_cap.read()
    videoRGB=cv.cvtColor(video_data,cv.COLOR_BGR2RGB)
    results=hands.process(videoRGB)


    if results.multi_hand_landmarks:
        for handLms in results.multi_hand_landmarks:
            for id,lm in enumerate(handLms.landmark):
                #print(id,lm)
                h,w,c=video_data.shape
                cx,cy=int(lm.x*w),int(lm.y*h)
                print(id,cx,cy)
                if(id==4):
                    cv.circle(video_data,(cx,cy),15,(255,0,255),cv.FILLED)


            mpDraw.draw_landmarks(video_data,handLms,mpHands.HAND_CONNECTIONS)


    cTime=time.time()
    fps=1/(cTime-pTime)
    pTime=cTime

    cv.putText(video_data,str(int(fps)),(10,70),cv.FONT_HERSHEY_DUPLEX,3,(0,255,255),4)

    cv.imshow("LIVE",video_data)
    if cv.waitKey(10)==ord("a"):
        break
video_cap.release()

