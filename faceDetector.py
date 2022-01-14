import cv2

face_cascade=cv2.CascadeClassifier("E:\Coding\PythonProjects\FaceDetector\haarcascade_frontalface_default.xml")

img=cv2.imread("E:\Coding\PythonProjects\FaceDetector\photo.jpg")

faces=face_cascade.detectMultiScale(img,
scaleFactor=1.1,minNeighbors=5)

for x,y,w,h in faces:
    img=cv2.rectangle(img, (x,y),(x+w,y+h),(0,255,0),  3)

resized=cv2.resize(img,(int(img.shape[1]/3),int(img.shape[0]/3)))

cv2.imshow('BW',resized)
cv2.waitKey(0)
cv2.destroyAllWindows()
 