"""Inference demo for YAMNet using RFCx format."""
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

    model = tf.saved_model.load('model')

    metadata_fn = model.signatures["metadata"]
    metadata = metadata_fn()
    print('metadata', metadata)

    score_fn = model.signatures["score"]
    print(score_fn)

    yamnet_classes = yamnet_model.class_names('yamnet_class_map.csv')

    print(yamnet_classes)

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

        waveform = tf.expand_dims(tf.expand_dims(tf.constant(waveform, dtype=tf.float32), 0), 2)

        scores = next(iter(score_fn(
            waveform=waveform,
            context_step_samples=tf.constant(int(params.PATCH_HOP_SECONDS * params.SAMPLE_RATE), dtype=tf.int64),
        ).values())).numpy()

        print(scores)

        # Scores is a matrix of (time_frames, num_classes) classifier scores.
        # Average them along time to get an overall classifier output for the clip.
        prediction = np.mean(scores[0], axis=0)
        # Report the highest-scoring classes and their scores.
        top5_i = np.argsort(prediction)[::-1][:5]
        print(file_name, ':\n' +
              '\n'.join('  {:12s}: {:.5f}'.format(yamnet_classes[i], prediction[i])
                        for i in top5_i))


if __name__ == '__main__':
    main(sys.argv[1:])
