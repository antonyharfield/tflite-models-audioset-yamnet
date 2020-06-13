"""Generate an RFCx format model from the original YAMNet with h5 model weights."""
from __future__ import division, print_function

import sys
import re

import tensorflow as tf
import numpy as np

import params
import yamnet as yamnet_model
from rfcx_frame import RfcxFrame

def main():

    # Load yamnet
    yamnet = yamnet_model.yamnet_frames_model(params)
    yamnet.load_weights('yamnet.h5')
    
    # Convert the model
    class_names = [re.sub(r'\ |\(|\)|,|-|\'', '', x.lower()) for x in yamnet_model.class_names('yamnet_class_map.csv')]
    
    frame = RfcxFrame(yamnet, params.SAMPLE_RATE, params.PATCH_WINDOW_SECONDS, class_names, 'pcm_s16le')
    tf.saved_model.save(frame, 'model', signatures={"score": frame.score, "metadata": frame.metadata})


if __name__ == '__main__':
    main()
