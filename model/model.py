"""Define the model."""

import tensorflow as tf
from tensorflow import keras

# from tensorflow.keras import layers
# from tensorflow.keras import activations
# from keras.layers import BatchNormalization
from tensorflow.keras import Model
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Activation, Dropout, Dense, Flatten, Dropout, BatchNormalization, GlobalAveragePooling2D, MaxPooling2D, Conv2D

def buil_model(is_training, image_size, params, classes = 3):
    
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
    assert IMG_SHAPE == (params.image_size, params.image_size, 3)

    if params.use_ResNet:
        base_model = tf.keras.applications.InceptionResNetV2(input_shape=IMG_SHAPE,
                                            include_top=False,
                                            weights='imagenet')
    else:
        base_model = tf.keras.applications.MobileNetV2(input_shape=IMG_SHAPE,
                                            include_top=False,
                                            weights='imagenet')

    print("[INFO] creating model...")
    print('[INFO] Base model is loaded ...')
    # -----------------------------------------------------------
    # MODEL: define the layers of the model
    # Show a summary of the model. Check the number of trainable parameters
    # must be true if the entire model is going to be trained
    base_model.trainable = params.base_train

    # build the model using Keras' Sequential API
    model = Sequential()
    # base model from MobileNet
    model.add(base_model)
  
    # global average layer (also faltten the output of MobileNet)
    model.add(GlobalAveragePooling2D())

    # first FC layer with droupout
    model.add(Dense(params.predic_layer_size))

    if params.use_batch_norm:
        model.add(BatchNormalization())

    if params.use_tanh:
        # activation_layer 
        model.add(Activation('tanh'))
    else:
        model.add(Activation('relu'))

    # drop out
    if params.use_dropout:
        model.add(Dropout(0.5))

    # prediction layer with 3 outputs
    model.add(Dense(classes, activation= 'linear'))
    # -----------------------------------------------------------
    # return the constructed network architecture
    return model


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
    image_size = (params.image_size, params.image_size, 3)
    # -----------------------------------------------------------
    # MODEL:
    # Compute the output distribution of the model and the predictions
    model = buil_model(is_training, image_size, params)
    print('[INFO] Final model is loaded ...')
    # TODO add Prediction: prediction = model(x, training=False)
    # Define loss and accuracy
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
        'train_RMSE' : tf.keras.metrics.RootMeanSquaredError(name='train_rmse'),
        'train_MSE' : tf.keras.metrics.MeanSquaredError(name='train_mse'),
        'train_MAE' : tf.keras.metrics.MeanAbsoluteError(name='train_mae'),

        'test_loss' : tf.keras.metrics.Mean(name='test_loss', dtype=tf.float32),
        'test_accuracy' :tf.keras.metrics.RootMeanSquaredError(name='test_accuracy')
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