# import the necessary packages
import numpy as np
import cv2

def reorder(myPoints):

    myPoints = myPoints.reshape((4, 2))
    #myPoints =  [[177 160]
    #            [317 161]
    #            [323 195]
    #            [180 193]]
    #print ("myPoints = ",myPoints)
    myPointsNew = np.zeros((4, 2), dtype=np.int32)
    #myPointsNew =  [[0 0]
    #               [0 0]
    #               [0 0]
    #               [0 0]]
    add = myPoints.sum(1) #1 means working along the row
    #add =  [337 478 518 373]
    #print ("add = ",add)

    myPointsNew[0] = myPoints[np.argmin(add)]
    myPointsNew[2] =myPoints[np.argmax(add)]

    diff = np.diff(myPoints, axis=1)
    myPointsNew[3] =myPoints[np.argmin(diff)]
    myPointsNew[1] = myPoints[np.argmax(diff)]
    #print ("diff = ", diff)

    #print("myPointsNew = ", myPointsNew)
    return myPointsNew

