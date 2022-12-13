import sys
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

def counter(image):
    _, file_name = os.path.split( image )
    status = 0
    image = cv2.imread(image)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    blur = cv2.GaussianBlur(gray, (11, 11), 0)
    canny = cv2.Canny(blur, 30, 150, 3)
    dilated = cv2.dilate(canny, (1, 1), iterations=0)
    
    (cnt, hierarchy) = cv2.findContours(
        dilated.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    cv2.drawContours(rgb, cnt, -1, (0, 255, 0), 2)
    
    
    print("coins in the image : ", len(cnt))

    # cv2.imshow("Top 'k' features", rgb)

    status = 1
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(rgb, str(len(cnt)), (0,100), font, 3, (0, 255, 0), 2, cv2.LINE_AA)
    cv2.imwrite( f"./processed_files/{file_name}", rgb )
    return status
