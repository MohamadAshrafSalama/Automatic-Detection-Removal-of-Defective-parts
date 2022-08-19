import cv2 as cv
import numpy as np
import serial

import time


import matplotlib.pyplot as plt

import time
#
#taking video caputer 0-> webcam 1-> 2nd cam

kernel = np.ones((5,5),np.uint8)
vid=cv.VideoCapture(0)

while(True):
    ret,fram=vid.read()
    fram=fram[200:350,130:640]
    cv.imshow("vid",fram)

    if cv.waitKey(1) & 0xFF == ord('q'):

        cv.destroyAllWindows()
        break

#taking the last fram to work on it
balls=fram




#conv it to hsv better control
hsv=cv.cvtColor(balls,cv.COLOR_BGR2HSV)

#chosing the color
lower_color = np.array([0,161,70], dtype=np.uint8)
upper_color = np.array([180,255,255], dtype=np.uint8)

#making a mask to block all other colors and applying it
mask = cv.inRange(hsv, lower_color, upper_color)
res = cv.bitwise_and(balls,balls, mask= mask)

#filling the gapes
dilation = cv.dilate(res,kernel,iterations = 5)

#showing the result of the masking


mask2=cv.cvtColor(dilation,cv.COLOR_BGR2HSV)
mask2 = cv.inRange(mask2, lower_color, upper_color)
mask2 = cv.dilate(mask2,kernel,iterations = 5)

#cv.imwrite("deez.jpg",mask2)
#
# cv.imshow("dd2",mask2)
cv.imwrite("sample.jpg",mask2)
# cv.waitKey(0)

#getting the countours of the mask  and drawing it
img = cv.imread("sample.jpg")

img_grey = cv.cvtColor(img,cv.COLOR_BGR2GRAY)

thresh = 100

ret,thresh_img = cv.threshold(img_grey, thresh, 255, cv.THRESH_BINARY)

contours, hierarchy = cv.findContours(thresh_img, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)


img_contours = np.zeros(img.shape)

cv.drawContours(fram, contours, -1, (0,255,0), 3)

# cv.imshow("lol",fram)
#
# cv.waitKey(0)

#getting the centers and printing it
for i in contours:
    M = cv.moments(i)
    if M['m00'] != 0:
        cx = int(M['m10']/M['m00'])
        cy = int(M['m01']/M['m00'])
        cv.drawContours(fram, [i], -1, (0, 255, 0), 2)
        cv.circle(fram, (cx, cy), 7, (0, 0, 255), -1)
        cv.putText(fram, "Defected bottle", (cx - 20, cy - 20),
                   cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
        cy=150-cy
        break

print(f"x: {cx} y: {cy}") #final output

cv.imshow("final",fram)
cv.waitKey(0)


#to serial

serialcomm = serial.Serial('COM4', 9600)

serialcomm.timeout = 1


serialcomm.write(cx)