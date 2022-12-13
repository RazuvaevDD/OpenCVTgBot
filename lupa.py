# from __future__ import print_function
import os
import sys
import numpy as np
import cv2
import glob
# from matplotlib import pyplot as plt
# from common import splitfn

def lupa( image ):
    _, file_name = os.path.split( image )
    status = 0
    img_names_undistort = [img for img in glob.glob( image )]
    new_path = "/путь для сохранения обработанных изображений/"

    camera_matrix = np.array([[1.26125746e+03, 0.00000000e+00, 9.40592038e+02],
                            [0.00000000e+00, 1.21705719e+03, 5.96848905e+02],
                            [0.00000000e+00, 0.00000000e+00, 1.00000000e+00]]);
    dist_coefs = np.array([-0.49181345,  0.25848255, -0.01067125, -0.00127517, -0.01900726]);

    i = 0

    #for img_found in img_names_undistort:
    while i < len(img_names_undistort):
        img = cv2.imread(img_names_undistort[i])
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        h,  w = img.shape[:2]
        newcameramtx, roi = cv2.getOptimalNewCameraMatrix(camera_matrix, dist_coefs, (w, h), 1, (w, h))

        dst = cv2.undistort(img, camera_matrix, dist_coefs, None, newcameramtx)

        dst = cv2.cvtColor(dst, cv2.COLOR_BGR2RGB)

        # crop and save the image
        x, y, w, h = roi
        dst = dst[y:y+h-50, x+70:x+w-20]

        i = i + 1
        status = 1
    cv2.imwrite( f"./processed_files/{file_name}", dst )
    return status
