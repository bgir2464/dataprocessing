import os
from preprocessing.detect_shapes import detect_circle
import tensorflow as tf
import numpy as np
import cv2
import tifffile as tiff
from polygon_generation import *
from skimage.transform import resize

from preprocessing.preprocessing import get_files

INPUT_DIR = "white_forms_resized"
REAL_DIR = "forms_mirrored_resized"
IMG_WIDTH = 256
IMG_HEIGHT = 256

DEBUG=1
def mat2gray(img):
    A = np.double(img)
    out = np.zeros(A.shape, np.double)
    normalized = cv2.normalize(A, out, 1.0, 0.0, cv2.NORM_MINMAX)
    return out




def random_noise(image, mode='gaussian', seed=None, clip=True, **kwargs):
    image = mat2gray(image)
    mode = mode.lower()
    if image.min() < 0:
        low_clip = -1
    else:
        low_clip = 0
    if seed is not None:
        np.random.seed(seed=seed)

    if mode == 'gaussian':
        noise = np.random.normal(kwargs['mean'], kwargs['var'] ** 0.5,
                                 image.shape)
        out = image + noise
    if clip:
        out = np.clip(out, low_clip, 1.0)
    img1 = np.uint8(out * 255)
    im=cv2.merge((img1,img1,img1))
    return im


def load(image_files):
  input_image = tf.io.read_file(image_files[0])
  real_image = tf.io.read_file(image_files[1])
  input_image = tf.image.decode_jpeg(input_image)
  real_image=tf.image.decode_jpeg(real_image)
  input_image = tf.cast(input_image, tf.float32)
  real_image = tf.cast(real_image, tf.float32)

  return input_image, real_image


def resize(input_image, real_image, height, width):

    input_image = tf.image.resize(input_image, [height, width],
                                method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    real_image = tf.image.resize(real_image, [height, width],
                               method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)

    return input_image, real_image

def random_crop(input_image, real_image):
    stacked_image = tf.stack([input_image, real_image], axis=0)
    cropped_image = tf.image.random_crop(
      stacked_image, size=[2, IMG_HEIGHT, IMG_WIDTH, 3])

    return cropped_image[0], cropped_image[1]


def normalize(input_image, real_image):

    input_image = (input_image / 127.5) - 1
    real_image = (real_image / 127.5) - 1

    return input_image, real_image



@tf.function()
def random_jitter(input_image, real_image):

    # resizing to 286 x 286 x 3
    input_image, real_image = resize(input_image, real_image, 286, 286)

    # randomly cropping to 256 x 256 x 3
    input_image, real_image = random_crop(input_image, real_image)

    if tf.random.uniform(()) > 0.5:
      # random mirroring
      input_image = tf.image.flip_left_right(input_image)
      real_image = tf.image.flip_left_right(real_image)

    return input_image, real_image

input_dir="white_forms_resized"
real_dir="forms_mirrored_resized"
# train_dataset = tf.data.Dataset.list_files(input_dir+"/*.jpg")
#
#
# print(len(train_dataset))


def load_image_train(image_file):
  input_image, real_image = load(image_file)
  input_image, real_image = random_jitter(input_image, real_image)
  input_image, real_image = normalize(input_image, real_image)

  return input_image, real_image

def noised_polygon_mask(height,width,radius,no_v):
    layer1 = np.zeros((width, height, 1))
    img = random_noise(layer1, 'gaussian', mean=0.5, var=0.1)

    no_of_polygons = randint(1, 8)
    polygons = []
    for i in range(0, no_of_polygons):
        points = polygon_points(height, width, radius, no_v)
        polygons.append(points)


    mask = np.zeros((img.shape), np.uint8)

    channel_count = img.shape[2]  # i.e. 3 or 4 depending on your image
    ignore_mask_color = (255,255,255)
    cv2.fillPoly(mask, polygons, ignore_mask_color)
    # from Masterfool: use cv2.fillConvexPoly if you know it's convex

    # apply the mask
    masked_image = cv2.bitwise_and(img, mask)


    ## (3) do bit-op
    # dst = cv2.bitwise_and(croped, croped, mask=mask)

    return masked_image

def add_mask(picture,mask):
    ret, mask2 = cv2.threshold(mask,0,255, cv2.THRESH_BINARY)
    # mask2 = cv2.bitwise_not(mask2)

    img2_fg = cv2.bitwise_and(mask[:,:,0], mask[:,:,0], mask=mask2[:,:,0])
    img2_fg_1 = cv2.bitwise_and(mask[:, :, 1], mask[:, :, 1], mask=mask2[:, :, 1])
    img2_fg_2 = cv2.bitwise_and(mask[:, :, 2], mask[:, :, 2], mask=mask2[:, :, 2])
    img2=cv2.merge((img2_fg,img2_fg_1,img2_fg_2))

    img1_bg = cv2.add(picture[:,:,0], img2[:,:,0])
    img2_bg = cv2.add(picture[:, :, 1], img2[:, :, 1])
    img3_bg = cv2.add(picture[:, :, 2], img2[:, :, 2])
    # img1_bg2 = cv2.add(picture, img2[:, :, 1])
    # img1_bg3 = cv2.add(picture, img2[:, :, 2])
    # dst=cv2.bitwise_and(picture,img2)
    img2_fg = cv2.bitwise_and(picture[:, :, 0], picture[:, :, 0], mask=img2[:, :, 0])
    img2_fg_1 = cv2.bitwise_and(picture[:, :, 1], picture[:, :, 1], mask=img2[:, :, 1])
    img2_fg_2 = cv2.bitwise_and(picture[:, :, 2], picture[:, :, 2], mask=img2[:, :, 2])

    imgf = cv2.merge((img1_bg, img2_bg, img3_bg))





def check_image(inp):
    im = cv2.imread(inp)
    # im2=tf.io.read_file(inp,dt)
    tf.keras.preprocessing.image.array_to_img(im)



def add_polygon(im):
    w, h, c = im.shape
    # resize(im,im,256,256)
    w, h, c = im.shape
    x,y,r=detect_circle(im)
    sides=randint(3,8)
    layer1 = noised_polygon_mask(h, w, r, sides)
    img2gray = cv2.cvtColor(layer1, cv2.COLOR_BGR2GRAY)
        # show_image(img2gray)
    ret, mask = cv2.threshold(img2gray, 10,255,cv2.THRESH_BINARY)
    mask_inv = cv2.bitwise_not(mask)

        # show_image(mask_inv)
    img1_bg = cv2.bitwise_and(im, im, mask=mask_inv)
    img2_fg = cv2.bitwise_and(layer1, layer1, mask=mask)

        # show_image(img1_bg)
        # show_image(img2_fg)
    dst = cv2.add(img1_bg, img2_fg)

#
# def add_mask(image,mask):
#
    return dst




def show_image(image):

    cv2.namedWindow('img', cv2.WINDOW_NORMAL)
    cv2.imshow('img', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

DEBUG=1

if DEBUG==1:
    inp="D:\\Uni\\Licenta\\data\\google_images\\ancient_plate\\ancient plate_1.jpeg"
    input="D:\\Uni\\Licenta\\data\\report_2012_without\\213438_381908.tiff"

    im = cv2.imread(inp)
    im = cv2.resize(im, (256,256))
    inp=add_polygon(im)

    files = get_files("D:/Uni/Licenta/data/google_images/porcelain_plate")

    #TO DO
    # resize pictures without loss of information
    #
    # save test pictures

    for f in files:
        try:
            im = cv2.imread(f)
            im = cv2.resize(im, (256, 256))
            inp = add_polygon(im)
            show_image(inp)
        except Exception:
            pass
        # print(f)
    # print(type(inp[0][0]))
    # check_image(inp)
    # im=cv2.imread("D:\\Uni\\Licenta\\data\\report_2012_without\\213438_381908.tiff")
    # im = cv2.resize(im, (256, 256), interpolation=cv2.INTER_AREA)
    # w,h,c=im.shape
    # # print(type(im[0][0][0]))
    # cv2.namedWindow('img', cv2.WINDOW_NORMAL)
    # layer1 =noised_polygon_mask(h,w,w//9,8)
    # im=add_mask(im,layer1)
    # cv2.imshow('img', layer1)
    # cv2.waitKey(0)
    # cv2.imshow('img', im)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()