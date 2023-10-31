import cv2
import numpy 


'''
cap = cv2.VideoCapture(0)
#img = cv2.imread("static/python/2.jpg")
address = "http://192.168.43.1:9000/video"
cap.open(address)
sucess,imgWebcam = cap.read()
#print ("imgWebcam= ",imgWebcam)
cv2.imshow('imgWebcam',imgWebcam)
cv2.waitKey(0)
cv2.destroyAllWindows()
'''
def ss(q):
    return q+1

def hkm(x,y):
    ss(x)
    return x+y

print (hkm(2,4))


