import cv2
import numpy as np
from os import listdir
from os.path import isfile, join

data_path='C:/Users/Soumya/Desktop/dataset1'
onlyfiles= [f for f in listdir(data_path) if isfile(join(data_path))]

Training_Data, Labels= [], []

for i, files in enumerate(onlyfiles):
    image_path=data_path + onlyfiles[i]
    images = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    Training_Data.append(np.asarray(images, dtype=np.uint))
    Labels.append(i)

Labels = np.asarray(Labels, dtype=np.int)

#Local Binary Pattern Histogram Face Recognizer
model=cv2.face.LBPHFaceRecognizer_create()

model.train(np.asarray(Training_Data),np.asarray(Labels))

print('Dataset model- Training completed')

#Detection Code
face_classifier = cv2.CascadeClassifier('C:/Users/Soumya/AppData/Local/Programs/Python/Python37/Lib/site-packages/cv2/data')

def face_detector(img,size=0.5):
     gray= cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
     faces= face_classifier.detectMultiScale(gray,1.3,5) #Scaling Factro

    if face is():
        return img,[]

    for(x,y,w,h) in faces:
        cv2.rectangle(img, (x,y),(x+w,y+h),(0,255,0),2)
        #Region of Interest
        roi= img[y:y+h, x:x+w]
        #To resize
        roi= cv2.resize(roi,(200,200))

        return img, roi

    capt= cv2.VideoCapture(0)

    while True:
        ret, frame= capt.read()
        image, face= face_detector(frame)

        try:
            face=cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
            result= model.predict(face)

        if result[1] < 500:
            confidence = int(100*(1-(result)/300))

        if confidence > 82:
            cv2.putText(image, 'Your name/ user name', (250,450), cv2.FONT_HERSHEY_COMPLEX,1,(255,255,255),2)
            cv2.imshow('Face Cropper', image)

        else:
            cv2.putText(image, 'Unknown', (250,450),cv2.FONT_HERSHEY_COMPLEX,1,(0,0,255),2)
            cv2.imshow('Face Cropper',image)

        except:
             cv2.putText(image, 'face not found',(250,450),1,(255,0,0),2)
             cv2.imshow('Face Cropper', image)
             pass

        if cv2.waitKey(1)==13:
            break

    capt.release()
    cv2.destroyAllWindows()





