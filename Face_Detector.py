import cv2
from random import randrange

trained_face_data = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

#img =cv2.imread('RDJ.png')
#to capture video from web cam
webcam=cv2.VideoCapture(0)

#iterate forever over frames
while True:
    ##read the current frame
    successful_frame_read , frame=webcam.read()
    #must convert to grayscale
    grayscale_img = cv2.cvtColor(frame ,cv2.COLOR_BGR2GRAY)
    face_coordinates= trained_face_data.detectMultiScale(grayscale_img)
    for (x,y,w,h) in face_coordinates:
      cv2.rectangle(frame,(x,y),(x+w,y+h),(randrange(256),randrange(256),randrange(256)),2)

    cv2.imshow('vizzp Face Detector',frame)
    key = cv2.waitKey(1)
 
    if key==81 or key==113:
          break

    #release the video capture object
webcam.release()
print("code completed")