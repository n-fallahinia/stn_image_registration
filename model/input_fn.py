"""Create the input data pipeline using `tf.data`"""
import tensorflow as tf
from utils.data import *

def preprocess_data(raw_image, aligned_image, size_w, size_h, label_size):
    """ preprocessing images """   

    raw_image_string = tf.io.read_file(raw_image)
    raw_image = load_image(raw_image_string, size_w, size_h, label_size, raw_image_flag = True)
    raw_image = augment_image(raw_image)

    aligned_image_string = tf.io.read_file(aligned_image)
    aligned_image = load_image(aligned_image_string, size_w, size_h, label_size, raw_image_flag = False)
    aligned_image = augment_image(aligned_image)

    return raw_image, aligned_image


def load_image(image_string, size_w, size_h, label_size, raw_image_flag):
    """ decoding the images """   

    image = tf.image.decode_jpeg(image_string, channels=3)
    image = tf.image.convert_image_dtype(image, tf.float32)
    # image = tf.expand_dims(image[:,:,1], 2) # only the green channel is being used
    if raw_image_flag:
        image = tf.image.resize(image, [size_w, size_h]) 
    else:
        image = tf.image.resize(image, [label_size, label_size]) 

    return image

def augment_image(image):
    """ augmenting the images """   

    # image = tf.image.adjust_saturation(image,0.5)
    image = tf.clip_by_value(image, 0.0, 1.0)

    return image

def input_fn(is_training, filenames_raw, filenames_aligned, params = None):
    """Input function for the SIGNS dataset.
    The filenames have format "{img}_{id}.jpg",For instance: "data_dir/img_0004.jpg".
    Args:
        is_training: (bool) whether to use the train or test pipeline. At training, we shuffle the data and have multiple epochs
        filenames: (list) filenames of the images, as ["data_dir/{img}_{id}.jpg"...]
        params: (Params) contains hyperparameters of the model (ex: `params.num_epochs`)
    """
    num_samples = len(filenames_raw)
    assert len(filenames_raw) == len(filenames_aligned), "raw and aligned images should have same length"

    # Create a Dataset serving batches of images and labels
    preproc_fn = lambda f, l: preprocess_data(f, l, params.image_size_w, params.image_size_h, params.label_size)

    if is_training:
        dataset = (tf.data.Dataset.from_tensor_slices((filenames_raw, filenames_aligned))
        .shuffle(buffer_size = num_samples)  # whole dataset into the buffer ensures good shuffling
        .map(preproc_fn, num_parallel_calls=params.num_parallel_calls)
        .batch(params.batch_size)
        .prefetch(tf.data.experimental.AUTOTUNE)  # make sure you always have one batch ready to serve (can also use one 1)
    )
    else:
        dataset = (tf.data.Dataset.from_tensor_slices((filenames_raw, filenames_aligned))
        .map(preproc_fn)
        .batch(params.batch_size)
        .prefetch(1)  # make sure you always have one batch ready to serve
    )

    return dataset

