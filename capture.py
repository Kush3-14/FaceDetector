import cv2
import pandas
from datetime import datetime

c=0
first_frame=None
status_list=[None,None]
times=[]
df=pandas.DataFrame(columns=["Start","End"])

video=cv2.VideoCapture(0)

while True:
    c+=1
    check, frame = video.read()
    status=0
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    img=cv2.GaussianBlur(img,(21,21),0)

    if first_frame is None:
        first_frame=img
        continue
    
    delta_frame=cv2.absdiff(first_frame,img)

    thres_frame=cv2.threshold(delta_frame,30,255,cv2.THRESH_BINARY)[1]

    (cnts,_)=cv2.findContours(thres_frame.copy(),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)


    for contour in cnts:
        if cv2.contourArea(contour)<10400:
            continue
        status=1
        (x,y,w,h)=cv2.boundingRect(contour)
        cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),2)
    status_list.append(status)
    if status_list[-1]==1 and status_list[-2]==0:
        times.append(datetime.now())
    if status_list[-1]==0 and status_list[-2]==1:
        times.append(datetime.now())
    
    
    cv2.imshow("color",frame)
    cv2.imshow("delta",delta_frame)
    cv2.imshow("thresh",thres_frame)

    key=cv2.waitKey(1)

    if key==ord("x"):
        if status==1:
            times.append(datetime.now())
        break
    print(status)

for i in range(0,len(times),2):
    df=df.append({"Start":times[i],"End":times[i+1]},ignore_index=True)

df.to_csv("E:\Coding\PythonProjects\FaceDetector\Times.csv ")
print(c)
video.release()
cv2.destroyAllWindows()