import struct

from preprocessing import *
from polygon_generation import *
import tifffile as tiff
from PIL import Image
dir_name="D:\\Uni\\Licenta\\data\\report_2012_with"
input_train="D:\\Uni\\Licenta\\data\\tarr2012_input_train"
output_train="D:\\Uni\\Licenta\\data\\tarr2012_output_train"

input_test="D:\\Uni\\Licenta\\data\\tarr2012_input_test"
from utils import *



def picture_polygons(image,height,width,no_v):
    image=generate_polygon(image,height,width,width//11,no_v)
    if DEBUG==1:
        # cv2.imwrite("noised_image.tiff",image)
        tiff.imsave('new.tiff', image)
        # tf.io.write_file("noised_image.tiff",image)

files=get_files(dir_name)
file=cv2.imread(files[0])

for file in files:
    im = tiff.imread(file)
    if len(im.shape)==3:
        path = file.split("\\")
        type=path[-1].split(".")
        input_image = tf.image.resize(im, [256, 256], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
        tf.keras.preprocessing.image.save_img(input_test+"\\"+type[-2]+".jpeg", input_image)


