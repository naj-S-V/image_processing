import numpy as np
import cv2 as cv
import glob
import matplotlib.pyplot as plt
import json

chessboardSize = (7,7)
frameSize = (1920,1080)
criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)
objp = np.zeros((chessboardSize[0] * chessboardSize[1], 3), np.float32)
objp[:,:2] = np.mgrid[0:chessboardSize[0],0:chessboardSize[1]].T.reshape(-1,2)
objpoints = [] # 3d point in real world space
imgpointsL = [] # 2d points in image plane.
imgpointsR = [] # 2d points in image plane.
imagesLeft = sorted(glob.glob('images/stereoLeft/*.png'))
imagesRight = sorted(glob.glob('images/stereoRight/*.png'))
for imgLeft, imgRight in zip(imagesLeft, imagesRight):
    imgL = cv.imread(imgLeft)
    imgR = cv.imread(imgRight)
    grayL = cv.cvtColor(imgL, cv.COLOR_BGR2GRAY)
    grayR = cv.cvtColor(imgR, cv.COLOR_BGR2GRAY)
    # Find the chess board corners
    retL, cornersL = cv.findChessboardCorners(grayL, chessboardSize, None)
    retR, cornersR = cv.findChessboardCorners(grayR, chessboardSize, None)
    # If found, add object points, image points (after refining them)
    if retL and retR == True:
        objpoints.append(objp)
        cornersL = cv.cornerSubPix(grayL, cornersL, (11,11), (-1,-1), criteria)
        imgpointsL.append(cornersL)
        cornersR = cv.cornerSubPix(grayR, cornersR, (11,11), (-1,-1), criteria)
        imgpointsR.append(cornersR)

""" retL, cameraMatrixL, distL, rvecsL, tvecsL = cv.calibrateCamera(objpoints, imgpointsL, frameSize, None, None)
heightL, widthL, channelsL = imgL.shape
newCameraMatrixL, roi_L = cv.getOptimalNewCameraMatrix(cameraMatrixL, distL, (widthL, heightL), 1, (widthL, heightL))
retR, cameraMatrixR, distR, rvecsR, tvecsR = cv.calibrateCamera(objpoints, imgpointsR, frameSize, None, None)
heightR, widthR, channelsR = imgR.shape
newCameraMatrixR, roi_R = cv.getOptimalNewCameraMatrix(cameraMatrixR, distR, (widthR, heightR), 1, (widthR, heightR))
flags = 0
flags |= cv.CALIB_FIX_INTRINSIC
criteria_stereo= (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)
retStereo, newCameraMatrixL, distL, newCameraMatrixR, distR, rot, trans, essentialMatrix, fundamentalMatrix = cv.stereoCalibrate(objpoints, imgpointsL, imgpointsR, newCameraMatrixL, distL, newCameraMatrixR, distR, grayL.shape[::-1], criteria_stereo, flags)
rodrigLeft = cv.Rodrigues(rvecsL[0])
cLeft = np.linalg.inv(rodrigLeft[0]) @ (-tvecsL[0])
cLeft = np.concatenate(cLeft, [1])
print(cLeft)
rodrigRight = cv.Rodrigues(rvecsR[0])
cRight = np.linalg.inv(rodrigRight[0]) @ (-tvecsR[0]) """

fundamentalMatrix = np.array([[-1.19975321e-08, -1.20581369e-06,  8.10150350e-04],[ 3.04336039e-06 , 5.88843356e-08, -1.39219298e-02],[-1.89773637e-03 , 1.22243249e-02,  1.00000000e+00]])
path = r'C:\Users\louis\Documents\Programmation\Traitement d\'images\Labo 3 & 4 - Projet -Stéréovision\scanL\0010.png'
img = cv.imread(path)
print(img)
# img = np.int64(img)
# lines = cv.computeCorrespondEpilines(img.reshape(-1, 1, 2), 2, fundamentalMatrix)
#print(lines)