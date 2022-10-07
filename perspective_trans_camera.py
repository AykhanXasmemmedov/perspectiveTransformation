import cv2
import numpy as np

cap=cv2.VideoCapture(0)
while True:
    ret,frame=cap.read()
    blur=cv2.GaussianBlur(frame,(5,5),0)
    gray=cv2.cvtColor(blur,cv2.COLOR_BGR2GRAY)
    ret,thresh=cv2.threshold(gray,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)

    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)  

    contours=sorted(contours,key=cv2.contourArea,reverse=True)
    cv2.drawContours(frame,contours,-1,(0,0,255),2)
    cv2.imshow('videotest',frame)
    for cnt in contours:
        perimeter=cv2.arcLength(cnt,True)
        approx=cv2.approxPolyDP(cnt,0.05*perimeter,True)
        if len(approx)==4:
            existing=True
            break
    
    minx=min(approx[:,:,0])
    maxx=max(approx[:,:,0])
    miny=min(approx[:,:,1])
    maxy=max(approx[:,:,1])  

    xleng=int(maxx-minx)
    yleng=int(maxy-miny)
    
    input=np.float32(approx)
    output=np.float32([[0,0],[0,yleng],[xleng,yleng],[xleng,0]])

    if len(input)==4 and len(output)==4:
        matrix=cv2.getPerspectiveTransform(input,output)
        perspective=cv2.warpPerspective(frame,matrix,(xleng,yleng))

        cv2.imshow('perspective',perspective)
    cv2.imshow('thresh',thresh)
    key=cv2.waitKey(7)
    if key==ord('q'):
        break
cap.release()
cv2.destroyAllWindows()

