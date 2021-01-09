#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#과제2


# In[57]:


#1-(1) 수동(마우스 클릭)으로 코너 검출
import numpy as np
import cv2


def draw_borderline(img, corners):
    copy_image = img.copy()
    for pt in corners:
        cv2.circle(copy_image, tuple(pt), 15, (0, 255, 255), -1, 1)
    cv2.line(copy_image, tuple(corners[0]), tuple(corners[1]), (0, 255, 255), 4, 5)
    cv2.line(copy_image, tuple(corners[1]), tuple(corners[2]), (0, 255, 255), 4, 5)
    cv2.line(copy_image, tuple(corners[2]), tuple(corners[3]), (0, 255, 255), 4, 5)
    cv2.line(copy_image, tuple(corners[3]), tuple(corners[0]), (0, 255, 255), 4, 5)
    return copy_image


def onMouse(event, x, y, flags, param):
    global points
    if event == cv2.EVENT_LBUTTONDOWN:
        for i in range(4):
            if cv2.norm(start_points[i] - (x, y)) < 20:
                drag_points[i] = True
                points = (x, y)
                break

    if event == cv2.EVENT_LBUTTONUP:
        for i in range(4):
            drag_points[i] = False

    if event == cv2.EVENT_MOUSEMOVE:
        for i in range(4):
            if drag_points[i]:
                dx = x - points[0]
                dy = y - points[1]
                start_points[i] += (dx, dy)
                cpy = draw_borderline(image, start_points)
                cv2.imshow('image', cpy)
                points = (x, y)
                break



image = cv2.imread('book.jpg')
print(image.shape)
h, w = image.shape[:2]

result_w = 250
result_h = 250

start_points = np.array([[50, 50], [50, h-50], [w-50, h-50], [w-50, 50]], np.float32)
end_points = np.array([[0, 0], [0, result_h-1], [result_w-1, result_h-1], [result_w-1, 0]], np.float32)

drag_points = [False, False, False, False]


borderline = draw_borderline(image, start_points)
cv2.namedWindow('image',cv2.WINDOW_NORMAL)

cv2.imshow('image', borderline)
cv2.setMouseCallback('image', onMouse)

while True:
    key = cv2.waitKey()
    if key == 13:
        break
    elif key == 27:
        cv2.destroyWindow('image')

# 투시 변환
perspective_transform = cv2.getPerspectiveTransform(start_points, end_points)
final_result = cv2.warpPerspective(image, perspective_transform, (result_w, result_h), flags=cv2.INTER_CUBIC)

# 결과 영상 출력
cv2.imshow('dst', final_result)
cv2.waitKey()
cv2.destroyAllWindows()


# In[51]:


#1-(2) 자동으로 코너검출

import cv2, sys
from matplotlib import pyplot as plt
import numpy as np

image = cv2.imread('book.jpg')
image_gray = cv2.imread('book.jpg', cv2.IMREAD_GRAYSCALE)


blur = cv2.GaussianBlur(image_gray, ksize=(13,13), sigmaX=0)
edged = cv2.Canny(blur, 10, 250)

cv2.imshow('Edged', edged)
cv2.waitKey(0)


corners=cv2.goodFeaturesToTrack(edged,4,0.01,300)

for i in corners:
    cv2.circle(image,tuple(i[0]),4,(0,0,255),2)
    
cv2.imshow('corner',image)
cv2.waitKey(0)
cv2.destroyAllWindows()


result_w = 250
result_h = 250


print(corners)

start=np.array([corners[0],corners[1],corners[2],corners[3]])
end_points = np.array([[0, 250], [250, 250], [0, 0], [250, 0]], np.float32)

#투시변환
perspective_transform = cv2.getPerspectiveTransform(start,end_points)
final_result = cv2.warpPerspective(image, perspective_transform, (result_w, result_h), flags=cv2.INTER_CUBIC)

# 결과 영상 출력
cv2.imshow('dst', final_result)
cv2.waitKey()
cv2.destroyAllWindows()

#참고 : https://thebook.io/006939/ch08/02-01/
#참고 : https://076923.github.io/posts/Python-opencv-23/
#참고 : https://youbidan.tistory.com/19

