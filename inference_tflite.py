# Copyright 2019 The TensorFlow Authors All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Inference demo for YAMNet using tflite."""
from __future__ import division, print_function

import sys

import numpy as np
import resampy
import soundfile as sf
import tensorflow as tf

import params
import yamnet as yamnet_model


def main(argv):
  assert argv

  # Load the TFLite model and allocate tensors.
  interpreter = tf.lite.Interpreter(model_path="yamnet.tflite")
  interpreter.allocate_tensors()
  inputs = interpreter.get_input_details()
  outputs = interpreter.get_output_details()

  yamnet_classes = yamnet_model.class_names('yamnet_class_map.csv')

  for file_name in argv:
    # Decode the WAV file.
    wav_data, sr = sf.read(file_name, dtype=np.int16)
    assert wav_data.dtype == np.int16, 'Bad sample type: %r' % wav_data.dtype
    waveform = wav_data / 32768.0  # Convert to [-1.0, +1.0]

    # Convert to mono and the sample rate expected by YAMNet.
    if len(waveform.shape) > 1:
      waveform = np.mean(waveform, axis=1)
    if sr != params.SAMPLE_RATE:
      waveform = resampy.resample(waveform, sr, params.SAMPLE_RATE)

    # Predict YAMNet classes.
    interpreter.set_tensor(inputs[0]['index'], np.expand_dims(np.array(waveform, dtype=np.float32), axis=0))
    interpreter.invoke()
    scores = interpreter.get_tensor(outputs[0]['index'])

    # Scores is a matrix of (time_frames, num_classes) classifier scores.
    # Average them along time to get an overall classifier output for the clip.
    prediction = np.mean(scores, axis=0)
    # Report the highest-scoring classes and their scores.
    top5_i = np.argsort(prediction)[::-1][:5]
    print(file_name, ':\n' + 
          '\n'.join('  {:12s}: {:.3f}'.format(yamnet_classes[i], prediction[i])
                    for i in top5_i))


if __name__ == '__main__':
  main(sys.argv[1:])
