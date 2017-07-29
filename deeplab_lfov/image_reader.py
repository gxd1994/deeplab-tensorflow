import os

import numpy as np
import tensorflow as tf

#IMG_MEAN = np.array((104.00698793,116.66876762,122.67891434), dtype=np.float32)

def read_labeled_image_list(data_dir, data_list):
    """Reads txt file containing paths to images and ground truth masks.
    
    Args:
      data_dir: path to the directory with images and masks.
      data_list: path to the file with lines of the form '/path/to/image /path/to/mask'.
       
    Returns:
      Two lists with all file names for images and masks, respectively.
    """
    f = open(data_list, 'r')
    images = []
    masks = []
    for line in f:
        image, mask = line.strip("\n").split(' ')
        images.append(data_dir + image)
        masks.append(data_dir + mask)
    return images, masks

def read_images_from_disk(input_queue, input_size, random_scale, image_mean): 
    """Read one image and its corresponding mask with optional pre-processing.
    
    Args:
      input_queue: tf queue with paths to the image and its mask.
      input_size: a tuple with (height, width) values.
                  If not given, return images of original size.
      random_scale: whether to randomly scale the images prior
                    to random crop.
      
    Returns:
      Two tensors: the decoded image and its mask.
    """
    img_contents = tf.read_file(input_queue[0])
    label_contents = tf.read_file(input_queue[1])
    
    # img = tf.image.decode_jpeg(img_contents, channels=3)
    # label = tf.image.decode_png(label_contents, channels=1)

    img = tf.image.decode_png(img_contents, channels=3)
    label = tf.image.decode_png(label_contents, channels=1)

    if input_size is not None:
        h, w = input_size
        if random_scale:
            scale = tf.random_uniform([1], minval=0.75, maxval=1.25, dtype=tf.float32, seed=None)
            h_new = tf.to_int32(tf.mul(tf.to_float(tf.shape(img)[0]), scale))
            w_new = tf.to_int32(tf.mul(tf.to_float(tf.shape(img)[1]), scale))
            new_shape = tf.squeeze(tf.pack([h_new, w_new]), squeeze_dims=[1])
            img = tf.image.resize_images(img, new_shape)
            label = tf.image.resize_nearest_neighbor(tf.expand_dims(label, 0), new_shape)
            label = tf.squeeze(label, squeeze_dims=[0]) # resize_image_with_crop_or_pad accepts 3D-tensor.
        img = tf.image.resize_image_with_crop_or_pad(img, h, w)
        label = tf.image.resize_image_with_crop_or_pad(label, h, w)
    # RGB -> BGR.
    img_r, img_g, img_b = tf.split(img,[1,1,1],2)
    img = tf.cast(tf.concat([img_b, img_g, img_r],2), dtype=tf.float32)
    # Extract mean.
    img -= image_mean 
    return img, label

class ImageReader(object):
    '''Generic ImageReader which reads images and corresponding segmentation
       masks from the disk, and enqueues them into a TensorFlow queue.
    '''

    def __init__(self, data_dir, data_list, input_size, random_scale, image_mean,coord):
        '''Initialise an ImageReader.
        
        Args:
          data_dir: path to the directory with images and masks.
          data_list: path to the file with lines of the form '/path/to/image /path/to/mask'.
          input_size: a tuple with (height, width) values, to which all the images will be resized.
          random_scale: whether to randomly scale the images prior to random crop.
          coord: TensorFlow queue coordinator.
        '''
        self.data_dir = data_dir
        self.data_list = data_list
        self.input_size = input_size
        self.coord = coord
        
        self.image_list, self.label_list = read_labeled_image_list(self.data_dir, self.data_list)
        self.images = tf.convert_to_tensor(self.image_list, dtype=tf.string)
        self.labels = tf.convert_to_tensor(self.label_list, dtype=tf.string)
        self.queue = tf.train.slice_input_producer([self.images, self.labels],
                                                   shuffle=input_size is not None) # Not shuffling if it is val.
        self.image, self.label = read_images_from_disk(self.queue, self.input_size, random_scale, image_mean) 

    def dequeue(self, num_elements, is_training = True):
        '''Pack images and labels into a batch.
        
        Args:
          num_elements: the batch size.
          
        Returns:
          Two tensors of size (batch_size, h, w, {3,1}) for images and masks.'''
        # if is_training:

        #     num_preprocess_threads = 16
        #     min_queue_examples = 50
        #     image_batch, label_batch = tf.train.shuffle_batch(
        #                                     [self.image, self.label],
        #                                    batch_size=num_elements,
        #                                     num_threads=num_preprocess_threads,
        #                                    capacity=min_queue_examples + 3 * num_elements,
        #                                     min_after_dequeue=min_queue_examples)
        # else:

        #     image_batch, label_batch = tf.train.batch([self.image, self.label],
        #                                             num_threads = 4,
        #                                             batch_size = num_elements,capacity = 100)


        image_batch, label_batch = tf.train.batch([self.image, self.label],
                                            num_threads = 20,
                                            batch_size = num_elements,capacity = 100)

        return image_batch, label_batch