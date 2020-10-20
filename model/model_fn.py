"""Define the model."""

import tensorflow as tf

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.models import Sequential
from tensorflow.keras import regularizers

from model.utils import get_initial_weights
from model.stn_layers import BilinearInterpolation

def buil_model(is_training, image_size, params, sampling_size =(240, 240), channels = 3):
    
    """Compute logits of the model (output distribution)
    Args:
        is_training: (bool) whether we are training or not
        inputs: (dict) contains the inputs of the graph (features, labels...)
        params: (Params) hyperparameters
    Returns:
        output: output of the model
    """
    IMG_SHAPE = image_size
    chanDim = -1
    assert IMG_SHAPE == (params.image_size_w, params.image_size_h, channels)

    # -----------------------------------------------------------
    # MODEL: define the layers of the model
    # Show a summary of the model. Check the number of trainable parameters
    # must be true if the entire model is going to be trained
    # THIS IS JUST THE LOCALIZATION NETWORK

    image = Input(shape=IMG_SHAPE)
    locnet = MaxPooling2D(pool_size=(2, 2))(image)
    locnet = Conv2D(5, kernel_size=5, activation='relu')(locnet)
    locnet = MaxPooling2D(pool_size=(2, 2))(locnet)

    # locnet = Conv2D(10, kernel_size=7, activation='relu')(locnet)
    # locnet = MaxPooling2D(pool_size=(2, 2))(locnet)

    locnet = Flatten()(locnet)
    locnet = Dense(100, activation='relu',
            kernel_regularizer = regularizers.l2(1e-3),
            activity_regularizer = regularizers.l2(1e-3))(locnet)
    weights = get_initial_weights(100)
    locnet = Dense(6, weights=weights)(locnet)

    x = BilinearInterpolation(sampling_size)([image, locnet])

    # ----------------------------------------------------------
    # return the constructed network architecture
    return Model(inputs=image, outputs=x)


def model_fn(mode, params, reuse=False):
    """Model function defining the graph operations.
    Args:
        mode: (string) can be 'train' or 'eval'
        params: (Params) contains hyperparameters of the model (ex: `params.learning_rate`)
        reuse: (bool) whether to reuse the weights
        model: the NailNet model
    Returns:
        model_spec: (dict) contains the graph operations or nodes needed for training / evaluation
    """
    is_training = (mode == 'train')
    image_size = (params.image_size_w, params.image_size_h, 3)
    # -----------------------------------------------------------
    # MODEL:
    # Compute the output distribution of the model and the predictions
    model = buil_model(is_training, image_size, params)
    print('[INFO] Final model is loaded ...')
    # TODO add Prediction: prediction = model(x, training=False)
    # Define loss and accuracy

    if params.use_msle:
        loss_object = tf.keras.losses.MeanSquaredLogarithmicError()

    elif params.use_klD:
        loss_object = tf.keras.losses.KLDivergence(reduction="auto")

    else:
        loss_object = tf.keras.losses.MeanSquaredError(reduction=tf.keras.losses.Reduction.AUTO)

    # Define training step that minimizes the loss with the Adam optimizer
    if is_training:
        if params.adam_opt:
            opt = tf.keras.optimizers.Adam(learning_rate=params.learning_rate)
        else:
            opt = tf.keras.optimizers.RMSprop(lr=params.learning_rate, momentum=params.bn_momentum)
    # -----------------------------------------------------------
    # METRICS AND SUMMARIES
    metrics = {
        'train_loss' : tf.keras.metrics.Mean(name='train_loss', dtype=tf.float32),
        'train_MSE' : tf.keras.metrics.MeanSquaredError(name='train_mse', dtype=tf.float32),
        'train_KLD' : tf.keras.metrics.KLDivergence(name='train_kld'),

        'test_MSE' :tf.keras.metrics.MeanSquaredError(name='test_mse', dtype=tf.float32),
        'test_KLD' :tf.keras.metrics.KLDivergence(name='test_kld')

    }
    # -----------------------------------------------------------
    # MODEL SPECIFICATION
    # Create the model specification and return it
    # It contains nodes or operations in the graph that will be used for training and evaluation
    model_spec = {}
    model_spec['model'] = model
    if is_training:
        model_spec['loss'] = loss_object
        model_spec['opt'] = opt
        model_spec['metrics'] = metrics

    return model_spec