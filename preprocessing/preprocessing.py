import glob
import cv2 as cv
import os
import tensorflow as tf
DEBUG=1
dir= "../forms_mirrored_resized"

INPUT_WIDTH=256
INPUT_HEIGHT=256

def get_files(dirname):
    types=("jpg","jpeg","png","tiff")
    files=[]
    for type in types:
        files.extend(glob.glob(dirname+"/*."+type))
    return files

def half_white(image):
    height, width, channels = image.shape
    cv.rectangle(image, (0, 0), (width//2, height), (255, 255, 255), -1)
    if DEBUG == 1:
        cv.imwrite("../test/test.jpg", image)
    return image


def mirror_left(image):
    height, width, channels = image.shape
    half_right = tf.slice(image, [0,width // 2, 0], [height, width // 2, channels])
    half_left = tf.reverse(half_right, [1])
    res = tf.concat( [half_left, half_right],1)
    res = tf.image.encode_png(res)
    if DEBUG == 1:

        tf.io.write_file("test/test_mirror2.jpg", res)
    return res

def resize(input_image,height, width):
    input_image = tf.image.resize(input_image, [height, width],method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    input_image = tf.image.encode_png(input_image)
    if DEBUG == 1:
        tf.io.write_file("test/test_resize.jpg", input_image)
    return input_image

def write_to_dir(image,filename,dir):
    path=os.path.join(dir,filename)
    try:
        os.stat(dir)
    except:
        os.mkdir(dir)
    tf.io.write_file(path,image)

def process_images(images):
    for image in images:
        read_image=cv.imread(image)
        mirrored=mirror_left(read_image)
        # resized_image=resize(read_image,INPUT_HEIGHT,INPUT_WIDTH)
        path=os.path.split(image)
        write_to_dir(mirrored,path[-1],dir)





# files=get_files("D:/Uni/Licenta/data/forms_resized")
# print(len(files))
# read_image=cv.imread(files[0])
# mirror_left(read_image)
# process_images(files)
