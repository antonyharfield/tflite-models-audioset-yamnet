import tensorflow as tf
from tensorflow.keras import Model, layers
import features as features_lib
import params
from yamnet import yamnet


def yamnet_frames_tflite_model(feature_params):
    num_samples = int(round(params.SAMPLE_RATE * 0.975))
    waveform = layers.Input(batch_shape=(1, num_samples))
    # Store the intermediate spectrogram features to use in visualization.
    spectrogram = features_lib.waveform_to_log_mel_spectrogram(
        tf.squeeze(waveform, axis=0), feature_params)
    patches = features_lib.spectrogram_to_patches(spectrogram, feature_params)
    predictions = yamnet(patches)
    frames_model = Model(name='yamnet_frames',
                         inputs=waveform, outputs=[predictions, spectrogram])
    return frames_model


def main():
    # Load the model and weights
    model = yamnet_frames_tflite_model(params)
    model.load_weights('yamnet.h5')

    # Convert the model
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    tflite_model = converter.convert()
    open("yamnet.tflite", "wb").write(tflite_model)


if __name__ == '__main__':
    main()
