import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import PIL.Image

def load_img(image_path):
  max_dim = 512
  img = tf.io.read_file(image_path)
  img = tf.image.decode_image(img, channels=3)#Detects the image to perform apropriate opertions
  img = tf.image.convert_image_dtype(img, tf.float32)#converts image to tensor dtype

  shape = tf.cast(tf.shape(img)[:-1], tf.float32)# Casts a tensor to float32.

  long_dim = max(shape)
  scale = max_dim / long_dim

  new_shape = tf.cast(shape * scale, tf.int32)

  img = tf.image.resize(img, new_shape)

  return img[tf.newaxis, :]

def imshow(image, title=None):
  if len(image.shape) > 3:
    image = tf.squeeze(image, axis=0)

  plt.imshow(image)
  if title:
    plt.title(title)

def clip_0_1(image):
  return tf.clip_by_value(image, clip_value_min=0.0, clip_value_max=1.0)

def tensor_to_image(tf_input):
  tf_input = tf_input*255
  tf_input = np.array(tf_input, dtype=np.uint8)
  if np.ndim(tf_input)>3:
    assert tf_input.shape[0] == 1
    tf_input = tf_input[0]
  return PIL.Image.fromarray(tf_input)