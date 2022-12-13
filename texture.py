#!/usr/bin/env python

from __future__ import print_function

import sys
import os
import numpy as np
import cv2 as cv

def texture(image):
    _, file_name = os.path.split( image )
    try:
        fn = image
    except:
        fn = 'starry_night.jpg'

    status = 0

    img = cv.imread(cv.samples.findFile(fn))
    if img is None:
        print('Failed to load image file:', fn)
        sys.exit(1)

    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    h, w = img.shape[:2]

    eigen = cv.cornerEigenValsAndVecs(gray, 15, 3)
    eigen = eigen.reshape(h, w, 3, 2)  # [[e1, e2], v1, v2]
    flow = eigen[:,:,2]

    vis = img.copy()
    vis[:] = (192 + np.uint32(vis)) / 2
    d = 12
    points =  np.dstack( np.mgrid[d/2:w:d, d/2:h:d] ).reshape(-1, 2)
    for x, y in np.int32(points):
        vx, vy = np.int32(flow[y, x]*d)
        cv.line(vis, (x-vx, y-vy), (x+vx, y+vy), (0, 0, 0), 1, cv.LINE_AA)
    # cv.imshow('input', img)
    # cv.imshow('flow', vis)
    status = 1
    cv.imwrite( f"./processed_files/{file_name}", vis )
    print('Done')
    return status
