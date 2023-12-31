from django.shortcuts import render, redirect
from django.views.decorators import gzip
from django.contrib.auth.forms import UserCreationForm
from django.contrib.auth import login as auth_login
from django.http import StreamingHttpResponse
import cv2
import numpy as np

#from .camera import VideoCamera
import threading

from subprocess import PIPE, run
# from static.python import Main

from .forms import SignUpForm






def signup(request):
    form = UserCreationForm()
    if request.method == 'POST':
        form = UserCreationForm(request.POST)
        if form.is_valid():
            user = form.save()
            auth_login(request, user)
            return redirect('home')

    return render(request, 'signup.html', {'form': form})
# convert the picture to Black and White
def bw(im):
    import cv2
    grayImage = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    # cv2.imshow("grayR",grayImage)
    # cv2.waitKey(0)

    ret, thresh = cv2.threshold(grayImage, 100, 255, cv2.THRESH_BINARY)
    im = cv2.bitwise_not(thresh)
    # cv2.imwrite("D:/My_course/Scripts/Spyder test/bw.jpg",im)
    im = cv2.dilate(im, None, iterations=10)
    # im = cv2.erode(im,None,iterations = 2)

    # cv2.imshow("bwR",im)
    # cv2.waitKey(0)
    return (im)


def reorder(myPoints):
    import numpy as np
    myPoints = myPoints.reshape((4, 2))
    # myPoints =  [[177 160]
    #            [317 161]
    #            [323 195]
    #            [180 193]]
    # print ("myPoints = ",myPoints)
    myPointsNew = np.zeros((4, 2), dtype=np.int32)
    # myPointsNew =  [[0 0]
    #               [0 0]
    #               [0 0]
    #               [0 0]]
    add = myPoints.sum(1)  # 1 means working along the row
    # add =  [337 478 518 373]
    # print ("add = ",add)

    myPointsNew[0] = myPoints[np.argmin(add)]
    myPointsNew[2] = myPoints[np.argmax(add)]

    diff = np.diff(myPoints, axis=1)
    myPointsNew[3] = myPoints[np.argmin(diff)]
    myPointsNew[1] = myPoints[np.argmax(diff)]
    # print ("diff = ", diff)

    # print("myPointsNew = ", myPointsNew)
    return myPointsNew


def foundcounterrec(img, imgAug, im, imgVideo):
    import cv2
    import numpy as np
    image = img.copy()
    # im = bw(im)
    contours, hierarchy = cv2.findContours(im, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    m = max(contours, key=cv2.contourArea)
    # print(contours)
    # cv2.drawContours(image, [m], -1, (0, 0, 255), 10)
    # for cont in contours:
    approx = cv2.approxPolyDP(m, 0.01 * cv2.arcLength(m, True), True)
    if len(approx) == 4:
        rect = reorder(approx)
        width = int(np.linalg.norm(rect[1] - rect[0]))
        hight = int(np.linalg.norm(rect[2] - rect[0]))

        # print ("width = ",width)
        # print("hight = ",hight)
        rect = np.float32(rect)
        # pts = np.float32([[0,0],[0,hT],[wT,hT],[wT,0]]).reshape(-1,1,2)
        pts = np.float32([[0, 0], [0, hT], [wT, hT], [wT, 0]]).reshape(-1, 1, 2)
        matrix = cv2.getPerspectiveTransform(pts, rect)
        dst = cv2.perspectiveTransform(pts, matrix)
        img2 = cv2.polylines(imgWebcam, [np.int32(dst)], True, (255, 0, 255), 3)
        temp = cv2.warpPerspective(imgVideo, matrix, (image.shape[1], image.shape[0]))
        # cv2.imshow("temp",temp)
        maskNew = np.zeros((imgWebcam.shape[0], imgWebcam.shape[1]), np.uint8)
        cv2.fillPoly(maskNew, [np.int32(dst)], (255, 255, 255))
        maskInv = cv2.bitwise_not(maskNew)
        # print (maskNew.shape)
        imgAug = cv2.bitwise_and(imgAug, imgAug, mask=maskInv)
        imgAug = cv2.bitwise_or(temp, imgAug)
    # temp = cv2.add(maskInv,temp)
    # cv2.imshow("imgAug",imgAug)
    return imgAug

class VideoCamera(object):
    def __init__(self):
        global video
        video = cv2.VideoCapture(0)

    def __del__(self):
        video.release()

    def gen(camera):
        frameCounter = 0
        imgTarget = cv2.imread('media/1Q7-black.jpg')
        imgTarget = cv2.resize(imgTarget, (480, 480))
        myVid = cv2.VideoCapture("media/Audi Q7.mp4")
        address = "http://192.168.0.143:9000/video"
        video.open(address)
        success, image = video.read()
        success, image = video.read()
        global hT, wT, cT
        success, imgVideo = myVid.read()
        #print (imgVideo)
        hT, wT, cT = imgTarget.shape

        # create the ORB Detector with 1000 features
        orb = cv2.ORB_create(nfeatures=4000)

        # Find the Key points and the Descriptors for the Target Image
        kp1, des1 = orb.detectAndCompute(imgTarget, None)

        while True:
            try:
                global imgWebcam

                sucess, imgWebcam = video.read()
                #print(imgWebcam)
                imgWebcam = cv2.resize(imgWebcam, (640, 480))
                imgAug = imgWebcam.copy()

                kp2, des2 = orb.detectAndCompute(imgWebcam, None)
                # imgWebcam = cv2.drawKeypoints(imgWebcam, kp2, None)

                # create bf matcher.
                bf = cv2.BFMatcher()
                matches = bf.knnMatch(des1, des2, k=2)

                good = []
                for m, n in matches:
                    # print("m = ",m)
                    # print ("n = ",n)
                    if m.distance < 0.75 * n.distance:
                        good.append(m)

                imgFeatures = cv2.drawMatches(imgTarget, kp1, imgWebcam, kp2, good, None, flags=2)

                if len(good) > 10:
                    detection = True
                    success, imgVideo = myVid.read()
                    imgVideo = cv2.resize(imgVideo, (wT, hT))
                    im = bw(imgWebcam)
                    imgAug = foundcounterrec(imgWebcam, imgAug, im, imgVideo)
                    frameCounter += 1
                    # cv2.imshow("imx", imx)
                    # cv2.imshow("imgWarp", imgWarp)

                    # cv2.imshow("img2", img2)

                # cv2.imshow("imgFeatures", imgFeatures)
                # cv2.imshow("imgTarget", imgTarget)
                # cv2.imshow("imgVideo", imgVideo)
                # cv2.imshow("final", imgAug)
                # cv2.waitKey(1)
                # return imgAug
                ret, jpeg = cv2.imencode('.jpg', imgAug)
                frame =  jpeg.tobytes()
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')
            except ValueError:
                ret, jpeg = cv2.imencode('.jpg', imgWebcam)
                frame =  jpeg.tobytes()
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')
            except cv2.error:
                ret, jpeg = cv2.imencode('.jpg', imgWebcam)
                frame =  jpeg.tobytes()
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')
            #frame = camera.get_frame()
            #yield (b'--frame\r\n'
            #       b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')


@gzip.gzip_page
def livefe(request):

    # cam = process()
    #cam = VideoCamera()
    return StreamingHttpResponse(VideoCamera.gen(VideoCamera()), content_type="multipart/x-mixed-replace;boundary=frame")
    #return render(request, 'streaming.html')



'''def streaming(request):
    print("HI")
	# run([sys.executable,'D:\\My_course\\Scripts\\2021\\augmanted reality\\Augmented Reality\\static\\python\\test.py',],shell= False,stdout=PIPE)
    exec(open('D:\\My_course\\Scripts\\2021\\augmanted reality\\Augmented Reality\\static\\python\\Main.py').read())
    return render(request, 'streaming.html')'''




