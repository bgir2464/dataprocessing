# detect circles in the image
import cv2 as cv
import numpy as np
from preprocessing import *
# from preprocessing import get_files

DEBUG=1

# def detect_blobs(image):
#     params = cv.SimpleBlobDetector_Params()
def detect_circle(image):
    cop = cv.cvtColor(image, cv.COLOR_BGR2GRAY)

    mask = np.zeros((image.shape[1], image.shape[0]))

    circles = cv.HoughCircles(cop, cv.HOUGH_GRADIENT, 1.2, 1000)
    if circles is not None:
    #     # convert the (x, y) coordinates   radius of the circles to integers
    #     circles = np.round(circles[0, :]).astype("int")
    #     # loop over the (x, y) coordinates and radius of the circles
    #     for (x, y, r) in circles:
    #         # draw the circle in the output image, then draw a rectangle
    #         # corresponding to the center of the circle
    #         cv.circle(mask, (x, y), r, (255, 255, 255), -1,cv.FILLED)
    #         # cv.rectangle(image, (x - 5, y - 5), (x + 5, y + 5), (0, 128, 255), -1)
    #     # show the output image
    # return mask
        circles = np.round(circles[0, :]).astype("int")
        (x, y, r) =circles[0]
        return x,y,r
    return None



# if DEBUG==1:
#     files=get_files("D:/Uni/Licenta/data/google_images/porcelain_plate")
#     # print(len(files))
#     cv.namedWindow('output', cv.WINDOW_NORMAL)
#     for i in range(0,50):
#         try:
#             im=cv.imread(files[i])
#             width = 256
#             height = 256
#             dim = (width, height)
#             im = cv.resize(im,dim )
#             detected=detect_circle(im)
#             cv.imshow("output", detected)
#             cv.waitKey(0)
#         except Exception as e:
#             print(e)
#
#     cv.destroyAllWindows()
