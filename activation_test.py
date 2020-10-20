
"""
Simple script to show the activation maps
for the STN and NailNet 
Navid Fallahinia - 09/15/2020
BioRobotics Lab
"""

import argparse
import os
from packaging import version

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input
import keract 
from quiver_engine import server
from model.stn_layers import BilinearInterpolation

parser = argparse.ArgumentParser()
parser.add_argument('--model_dir', default='./test',
                    help="Directory containing the model")

parser.add_argument('--image_dir', default="./data_3/test/raw_image/img_0000.jpg",
                    help="log directory for the trained model")

parser.add_argument('--mode', default='act_map', 
                    help="train or test mode")

parser.add_argument('--v', default=True,
                    help ='verbose mode')


if __name__ == '__main__':

    print("TensorFlow version: ", tf.__version__)
    assert version.parse(tf.__version__).release[0] >= 2, \
    "This notebook requires TensorFlow 2.0 or above."

    # Load the parameters from json file
    args = parser.parse_args()

    tf.debugging.set_log_device_placement(False)
    best_save_path = os.path.join(args.model_dir, "best_full_model_path")
    loaded_model = tf.keras.models.load_model(best_save_path)
    loaded_model.summary()

    img_path = args.image_dir
    raw_image_string = tf.io.read_file(img_path)
    raw_image = tf.image.decode_jpeg(raw_image_string, channels=3)
    raw_image = tf.image.convert_image_dtype(raw_image, tf.float32)
    raw_image = tf.image.resize(raw_image, [680, 460])
    raw_image = tf.clip_by_value(raw_image, 0.0, 1.0)
    image = tf.reshape(raw_image,(-1,680,460,3))

    print('=================================================')
    activations = keract.get_activations(loaded_model, image)
    [print(k, '->', v.shape, '- Numpy array') for (k, v) in activations.items()]

    if args.mode == 'act_map':
        keract.display_activations(activations, cmap="gray")
    elif args.mode == 'heat_map':
        keract.display_heatmaps(activations, image ,save=False)
    
    print('=================================================')
    if args.v:  
        json_path = os.path.join(args.model_dir, "affine_weights.json")
        weights = loaded_model.get_weights()
        print("[INFO] affine transformation params: ")
        [print("\t Theta -> ", w ) for(w) in weights[-1]]

    print('=================================================')

    out_image = loaded_model(image)
    IMG_SHAPE = (680,460,3)
    sampling_size = (240, 240)
    locnet = tf.constant(weights[-1])
    locnet = tf.reshape(locnet,(-1,6))
    x = BilinearInterpolation(sampling_size)([image, locnet])
    model = Model(inputs=image, outputs=x)
