#   Call Essential Libraries
from sys import flags
import numpy as np
import cv2 as cv
import glob

criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30,0.001)

#   Checkerboard Parameter Settings

checkerboardsize = (10,7)       # width / height
pattersize = (480,640)          # width / height
size_of_checkerboard_squares_mm = 24

objp = np.zeros((checkerboardsize[0] * checkerboardsize[1], 3), np.float32)
objp[:,:2] = np.mgrid[0:checkerboardsize[0],0:checkerboardsize[1]].T.reshape(-1,2)
objp = objp * size_of_checkerboard_squares_mm

objpoints = []      #   3d world point
imgpoints_L = []    #   2d image point of left image
imgpoints_R = []

images_left = sorted(glob.glob('images/left_images/*.png'))
images_right = sorted(glob.glob('images/right_images/*.png'))

for image_left, image_right in zip(images_left, images_right):
    img_L = cv.imread(image_left)
    img_R = cv.imread(image_right)
    img_gray_L = cv.cvtColor(img_L, cv.COLOR_BGR2GRAY)
    img_gray_R = cv.cvtColor(img_R, cv.COLOR_BGR2GRAY)

    ret_L, corneres_L = cv.findChessboardCorners(img_gray_L, checkerboardsize, None)
    ret_R, corneres_R = cv.findChessboardCorners(img_gray_R, checkerboardsize, None)

    if ret_L and ret_R == True:
        objpoints.append(objp)
        corneres_detail_L = cv.cornerSubPix(img_gray_L, corneres_L, (11,11), (-1,-1), criteria)
        imgpoints_L.append(corneres_detail_L)

        corneres_detail_R = cv.cornerSubPix(img_gray_R, corneres_R, (11,11), (-1,-1), criteria)
        imgpoints_R.append(corneres_detail_R)

        cv.drawChessboardCorners(img_L, checkerboardsize, corneres_detail_L, ret_L)
        cv.imshow('left image',img_L)
        cv.drawChessboardCorners(img_R, checkerboardsize, corneres_detail_R, ret_R)
        cv.imshow('right image',img_R)
        cv.waitKey(1000)

    elif ret_L and ret_R == False:
        print("Cannot find Corners in this image!!")

cv.destroyAllWindows()

#   Do Calibration
ret_L, camera_matrix_L, dist_L, rvecs_L, tvecs_L = cv.calibrateCamera(objpoints, imgpoints_L, pattersize, None, None)
height_L, width_L, channels_L = img_L.shape
newcameramatrix_L, roi_L = cv.getOptimalNewCameraMatrix(camera_matrix_L, dist_L, (width_L, height_L), 1, (width_L, height_L))


ret_R, camera_matrix_R, dist_R, rvecs_R, tvecs_R = cv.calibrateCamera(objpoints, imgpoints_R, pattersize, None, None)
height_R, width_R, channels_R = img_R.shape
newcameramatrix_R, roi_R = cv.getOptimalNewCameraMatrix(camera_matrix_R, dist_R, (width_R, height_R), 1, (width_R, height_R))

#   Stereo Vision Calibration

flags = 0
flags |= cv.CALIB_FIX_INTRINSIC

criteria_stereo= (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)

ret_Stereo, newcameramatrix_L, dist_L, newcameramatrix_R, dist_R, rotation, translation, essentialmatrix, fundamentalmatrix = cv.stereoCalibrate(objpoints, imgpoints_L, imgpoints_R, newcameramatrix_L, dist_L, newcameramatrix_R, dist_R, img_gray_L.shape[::-1],criteria_stereo, flags)


print("\nStereo Camera Calibrated\n", ret_Stereo)
print("\nLeft Camera Intrinsic matrix\n", newcameramatrix_L)
print("\nRight Camera Intrinsic matrix\n", newcameramatrix_R)

print("\nRotation between cameras\n", rotation)
print("\ntranslation between cameras\n", translation)
