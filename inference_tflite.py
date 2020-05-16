"""Inference demo for YAMNet using tflite."""
from __future__ import division, print_function

import sys

import numpy as np
import resampy
import soundfile as sf
import matplotlib.pyplot as plt

import params
import yamnet as yamnet_model
import yamnet_tflite as yamnet_tflite


def output(waveform, spectrogram, filename):
    plt.figure(figsize=(10, 8))
    # Plot the waveform.
    plt.subplot(2, 1, 1)
    plt.plot(waveform)
    plt.xlim([0, len(waveform)])
    # Plot the log-mel spectrogram (returned by the model).
    plt.subplot(2, 1, 2)
    plt.imshow(spectrogram.T, aspect='auto',
               interpolation='nearest', origin='bottom')
    plt.savefig(filename)
    plt.close()


def main(argv):
    assert argv

    # yamnet = yamnet_model.yamnet_frames_model(params)
    yamnet = yamnet_tflite.yamnet_frames_model(params)
    yamnet.load_weights('yamnet.h5')
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
        # Second output is log-mel-spectrogram array (used for visualizations).
        # (steps=1 is a work around for Keras batching limitations.)
        scores, spectrogram = yamnet.predict(
            np.reshape(waveform, [1, -1]), steps=1)

        # Scores is a matrix of (time_frames, num_classes) classifier scores.
        # Average them along time to get an overall classifier output for the clip.
        prediction = np.mean(scores, axis=0)
        # Report the highest-scoring classes and their scores.
        top5_i = np.argsort(prediction)[::-1][:5]
        print(file_name, ':\n' +
              '\n'.join('  {:12s}: {:.5f}'.format(yamnet_classes[i], prediction[i])
                        for i in top5_i))

        output(waveform, spectrogram, file_name + '.png')


if __name__ == '__main__':
    main(sys.argv[1:])
