import math
import random
from math import atan

import cv2
import numpy as np
# from preprocessing import DEBUG

from random import randint
from random import seed


DEBUG=1
W = 256
# seed(1)
radius=256//2
circle_x=256//2
circle_y=256//2

def circle_point(radius,circle_x,circle_y):
    alpha = 2 * math.pi * random.random()
    r = radius  # * math.sqrt(random.random())

    # calculating coordinates
    x = r * math.cos(alpha) + circle_x
    y = r * math.sin(alpha) + circle_y
    return x,y


def points_in_circle(radius,circle_x,circle_y,no):
    points=[]
    for i in range(0,no):
        x,y=circle_point(radius,circle_x,circle_y)
        points.append([x,y])

    print(points)

    vectors = sorted(points, key=lambda x: atan(x[1] / x[0]))
    print(vectors)
    return vectors


def convex_points(size,no_vertices):
    X=[]
    Y=[]
    for i in range(0,no_vertices):
        x=randint(0,size)
        y=randint(0,size)
        X.append(x)
        Y.append(y)
    X.sort()
    Y.sort()
    max_x=X[-1]
    min_x=X[0]
    max_y=Y[-1]
    min_y=Y[0]

    X.pop(0)
    X.pop(-1)
    Y.pop(0)
    Y.pop(-1)

    random.shuffle(X)
    index = random.randint(1, len(X)-1)
    print(index)
    x1 = X[index:]
    x2 = X[:index]

    random.shuffle(Y)
    index = random.randint(1, len(Y)-1)
    print(index)
    y1 = Y[index:]
    y2 = Y[:index]
    x1.insert(0,min_x)
    x2.insert(0,min_x)
    x1.append(max_x)
    x2.append(max_x)

    y1.insert(0,min_y)
    y1.append(max_y)
    y2.insert(0,min_y)
    y2.append(max_y)
    x1.sort()
    x2.sort()
    y1.sort()
    y2.sort()
    Xvec=[]
    Yvec=[]
    for i in range(0,len(x1)-1):
        dif=x1[i+1]-x1[i]
        Xvec.append(dif)

    for i in range(0,len(x2)-1):
        dif = x2[i] - x2[i+1]
        Xvec.append(dif)


    for i in range(0,len(y1)-1):
        dif=y1[i+1]-y1[i]
        Yvec.append(dif)



    for i in range(0,len(y2)-1):
        dif = y2[i] - y2[i+1]
        Yvec.append(dif)
    random.shuffle(Yvec)
    vectors=[]
    x_f=0
    y_f=0
    min_p_x=0
    min_p_y=0
    for i in range(0, len(Xvec)):
        vectors.append([Xvec[i],Yvec[i]])
    vectors = sorted(vectors, key=lambda x: atan(x[1] / x[0]))
    vect_lay=[]
    for i in range(0,len(vectors)):
        x_f+=x_f+vectors[i][0]
        y_f+=y_f+vectors[i][1]
        if x_f<min_p_x:
            min_p_x=x_f
        if y_f<min_p_y:
            min_p_y=y_f
        vect_lay.append([x_f,y_f])

    # vezi de ce nu le readuce in intervalul curespunzator
    #nu importeaza tensorflow, probabil de la prea multe console deschisea
    xShift = min_x - min_p_x
    yShift = min_y - min_p_y
    print(vect_lay)
    for i in range(0, len(vect_lay)):
        vect_lay[i][0]=vect_lay[i][0]+xShift
        vect_lay[i][1] = vect_lay[i][1]+yShift
    if DEBUG == 1:
        print(vect_lay)
    return vect_lay


def polygon_points(height,width,radius,no_vertices):
    ppt = np.array(points_in_circle(radius, width // 2, height // 2, no_vertices), np.int32)
    ch = cv2.convexHull(ppt)
    return ch



def generate_polygon(image,height,width,radius,no_vertices):
    line_type = 8
    no_of_polygons = randint(1, 5)
    polygons=[]
    for i in range(0,no_of_polygons):
        ppt=np.array(points_in_circle(radius,width//2,height//2,no_vertices),np.int32)
    # ppt = np.array([[W / 2, 2* W / 2], [W / 2, W / 2],
    #                 [ 2*W / 2, 2*W / 2], [-2*W/2,-2*W/2 ],
    #              ],np.int32)
        ch=cv2.convexHull(ppt)
        polygons.append(ch)
    cv2.fillPoly(image,polygons,(255, 255, 255),line_type)

    # cv2.drawContours(image,cnt,0,[0,0,0])
    return image


def generate_image(height,width):
    blank_image = np.zeros((height,width,3), np.uint8)
    image=generate_polygon(blank_image,height,width,width//2,20)
    if DEBUG==1:
        cv2.imwrite("test/image_test_polygon.jpg",image)
    return image

# DEBUG=1
#
# if DEBUG==1:
#
#     generate_image(256,256)

#convex_points(10,8)