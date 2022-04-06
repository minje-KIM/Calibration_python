#   Call Essential Libraries
import numpy as np
import cv2 as cv
import glob

criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30,0.001)

#   Checkerboard Parameter Settings

checkerboardsize = (10,7)   # width / height
patternsize = (480,640)     # width / height
size_of_checkerboard_squares_mm = 24

objp = np.zeros((checkerboardsize[0] * checkerboardsize[1], 3), np.float32)
objp[:,:2] = np.mgrid[0:checkerboardsize[0],0:checkerboardsize[1]].T.reshape(-1,2)
objp = objp * size_of_checkerboard_squares_mm

objpoints = []  #   3d world point
imgpoints = []  #   2d image point

#   Bring images from directory
images = glob.glob('images/*.png')

for image in images:
    img = cv.imread(image)
    img_gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    ret, corners = cv.findChessboardCorners(img_gray, checkerboardsize, None)

    if ret == True:
        
        objpoints.append(objp)
        corners_detail = cv.cornerSubPix(img_gray, corners, (11,11), (-1,-1), criteria)
        imgpoints.append(corners_detail)

        cv.drawChessboardCorners(img, checkerboardsize, corners_detail, ret)
        cv.imshow('img', img)
        cv.waitKey(1000)

    elif ret == False:
        print("Cannot find Corners in this image!!")

cv.destroyAllWindows()

#   Do calibration
ret, camera_matrix, dist, rvecs, tvecs = cv.calibrateCamera(objpoints, imgpoints, patternsize, None, None)

print("\nMono Camera Calibrated\n", ret)
print("\nIntrinsic Camera matrix\n", camera_matrix)
print("\nDistortion Parameters\n", dist)
#print("\nRotation Vectors\n", rvecs)
#print("\nTranslation Vectors\n", tvecs)

#   Get reprojection error
mean_error = 0

for i in range(len(objpoints)):
    imgpoints_reprojected, _ = cv.projectPoints(objpoints[i], rvecs[i], tvecs[i], camera_matrix, dist)
    error = cv.norm(imgpoints[i], imgpoints_reprojected, cv.NORM_L2)/len(imgpoints_reprojected)
    mean_error += error

print("\nTotal Error : {}".format(mean_error/len(objpoints)))
