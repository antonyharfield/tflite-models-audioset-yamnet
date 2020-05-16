"""Generate a tflite model from the original YAMNet with h5 model weights."""
from __future__ import division, print_function

import sys

import tensorflow as tf
import numpy as np

import params
import yamnet_tflite as yamnet_model


def main():

    # Load yamnet
    yamnet = yamnet_model.yamnet_frames_model(params)
    yamnet.load_weights('yamnet.h5')
    # yamnet.summary()

    # Convert the model
    converter = tf.lite.TFLiteConverter.from_keras_model(yamnet)
    converter.experimental_new_converter = True
    #converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS, tf.lite.OpsSet.SELECT_TF_OPS]
    tflite_model = converter.convert()
    open("yamnet.tflite", "wb").write(tflite_model)


if __name__ == '__main__':
    main()
