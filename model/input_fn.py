"""Create the input data pipeline using `tf.data`"""
import tensorflow as tf
from model.utils.data_util import *

def preprocess_data(image, force, size, use_random_flip, use_random_color, use_hsv):
    """ preprocessing images """   

    image_string = tf.io.read_file(image)
    image = load_image(image_string, size, use_hsv)
    image = augment_image(image, use_random_flip, use_random_color)

    return image, force


def load_image(image_string, size, use_hsv):
    """ decoding the images """   

    image = tf.image.decode_jpeg(image_string, channels=3)
    image = tf.image.convert_image_dtype(image, tf.float32) 
    image = tf.image.resize(image, [size, size]) 
    if use_hsv:
        image = tf.image.rgb_to_hsv(image)

    return image

def augment_image(image, use_random_flip, use_random_color):
    """ augmenting the images """   

    if use_random_flip:
        image = tf.image.random_flip_left_right(image)
        image = tf.image.random_flip_up_down(image)

    if use_random_color:
        image = tf.image.random_brightness(image, max_delta=32.0 / 255.0)
        image = tf.image.random_contrast(image, lower=0.5, upper=1.5)

    # image = tf.image.adjust_saturation(image,0.5)
    image = tf.clip_by_value(image, 0.0, 1.0)

    return image

def input_fn(is_training, filenames, labels, params = None):
    """Input function for the SIGNS dataset.
    The filenames have format "{img}_{id}.jpg",For instance: "data_dir/img_0004.jpg".
    Args:
        is_training: (bool) whether to use the train or test pipeline. At training, we shuffle the data and have multiple epochs
        filenames: (list) filenames of the images, as ["data_dir/{img}_{id}.jpg"...]
        labels: (np.array) corresponding np array of force lists
        params: (Params) contains hyperparameters of the model (ex: `params.num_epochs`)
    """
    num_samples = len(filenames)
    assert len(filenames) == len(labels), "Filenames and labels should have same length"

    # Create a Dataset serving batches of images and labels
    preproc_train_fn = lambda f, l: preprocess_data(f, l, params.image_size, params.use_random_flip, params.use_random_color, params.use_hsv)
    preproc_eval_fn = lambda f, l: preprocess_data(f, l, params.image_size, False, False, params.use_hsv)

    if is_training:
        dataset = (tf.data.Dataset.from_tensor_slices((filenames, labels))
        .shuffle(buffer_size = num_samples)  # whole dataset into the buffer ensures good shuffling
        .map(preproc_train_fn, num_parallel_calls=params.num_parallel_calls)
        .batch(params.batch_size)
        .prefetch(tf.data.experimental.AUTOTUNE)  # make sure you always have one batch ready to serve (can also use one 1)
    )
    else:
        dataset = (tf.data.Dataset.from_tensor_slices((filenames, labels))
        .map(preproc_eval_fn)
        .batch(params.batch_size)
        .prefetch(1)  # make sure you always have one batch ready to serve
    )

    # # Create reinitializable iterator from dataset
    # iterator = dataset.make_initializable_iterator()
    # images, labels = iterator.get_next()
    # iterator_init_op = iterator.initializer

    return dataset

