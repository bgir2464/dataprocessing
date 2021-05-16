# from google_images_download import google_images_download   #importing the library

from simple_image_download import simple_image_download as simp

response = simp.simple_image_download

response().download('google_images/','ancient plate', 1000)