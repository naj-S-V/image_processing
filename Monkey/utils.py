import cv2
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt


def calibrate(images):
    # termination criteria
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
    objp = np.zeros((6*7,3), np.float32)
    objp[:,:2] = np.mgrid[0:7,0:6].T.reshape(-1,2)

    # Arrays to store object points and image points from all the images.
    objpoints = [] # 3d point in real world space
    imgpoints = [] # 2d points in image plane.



    for img in images:
        #gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        # Find the chess board corners
        ret, corners = cv2.findChessboardCorners(img, (7,6),None)

        # If found, add object points, image points (after refining them)
        if ret == True:
            objpoints.append(objp)

            corners2 = cv2.cornerSubPix(img,corners,(11,11),(-1,-1),criteria)
            imgpoints.append(corners2)

            # Draw and display the corners
            img = cv2.drawChessboardCorners(img, (7,6), corners2,ret)
            # cv2.imshow('img',img)
            # cv2.waitKey(500)

    cv2.destroyAllWindows()

    print(imgpoints)
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, img.shape[::-1],None,None)
    return mtx, rvecs, tvecs



def findRedPoints(img):    
    '''
    trouver les points rouge d'une image
    '''

    img_hsv=cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # lower mask (0-10)
    lower_red = np.array([0,50,50])
    upper_red = np.array([10,255,255])
    mask0 = cv2.inRange(img_hsv, lower_red, upper_red)

    # upper mask (170-180)
    lower_red = np.array([170,50,50])
    upper_red = np.array([180,255,255])
    mask1 = cv2.inRange(img_hsv, lower_red, upper_red)

    # join my masks
    mask = mask0+mask1

    # select all points that's red
    output_img = img.copy()
    output_img[np.where(mask==0)] = 0
    redpoints = cv2.findNonZero(mask)
    try:
        for point in redpoints:
            point = point[0]
            cv2.circle(output_img, (point[0], point[1]), 2, (255,0,0))
    except:
        print("no red points")
    
    cv2.imshow('img',output_img)
    cv2.waitKey(100000)
    cv2.destroyAllWindows()
    
    
    return redpoints

def findInterest(redPoints, img):
    '''
    découper l'image avec 25 lignes
    trouver les points rouge sur ces lignes
    '''
    height, width, channels = img.shape
    #print(height)
    delta = height/25
    #print(delta)
    lines = []
    for i in range(26):
        newline = np.array([0, 1, - int(i*delta)])
        lines.append(newline)
        cv2.line(img,(0, int(i*delta)),(width, int(i*delta)),(255,0,0),1)
    
    points = [] 
    
    for line in lines:
        '''
        Le point  x'   appartient à la droite  l'   si et seulement si  xT @ l' = 0.
        faire un while pour stop quand on a trouvé
        '''
        found = False
        i = 0
        while not found and i < len(redPoints):
            point = redPoints[i]
            point = np.concatenate((point[0], [1]))
#             print("point: ", point)
#             print("line: ", line )
            if ((point.T @ line) == 0):
                points.append(point)
                found = True
                cv2.circle(img, (point[0], point[1]), 2, (255,255,0))
                #print(point)
            i += 1
    
    cv2.imshow('img',img)
    cv2.waitKey(100000)
    cv2.destroyAllWindows()
    return points

def findCorrespondence(leftPoints, imgR):
    '''
    faire les épilignes
    trouver les points rouge sur ces lignes
    '''
    rRedPoints = findRedPoints(imgR)
    left = []
    right = []
    
    
    for point in leftPoints:
        line = F @ point
#         line = np.array([int(line[0]), int(line[1]), int(line[2])])
        try:
            xi = int(-line[2] / line[0])
        except:
            xi = 0
        try:
            yi = int(-line[2] / line[1])
        except:
            yi = 0
        pt1 = (xi, 0)
        pt2 = (0, yi)
        
        cv2.line(imgR,pt2,pt1,(0,255,0),1)
        
        found = False
        i = 0
        while not found and i < len(rRedPoints):
            p = rRedPoints[i]
            p = np.concatenate((p[0], [1]))
#             print("point: ", p)
#             print("epiline: ", line )
            if (p.T @ line) <= 20000 and (p.T @ line) >= -20000:
#                 print("found")
#                 print(p)
#                 print(line)

                cv2.circle(imgR, (p[0], p[1]), 2, (0,255,255))
                
                left.append([point[0], point[1]])
                right.append([p[0], p[1]])
                found = True
            i += 1
        

        
    cv2.imshow('img',imgR)
    cv2.waitKey(100000)
    cv2.destroyAllWindows()

    return left, right