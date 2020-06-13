import tensorflow as tf
from tensorflow.keras import Model, layers
import features as features_lib
import features_tflite as features_tflite_lib
import params
from yamnet import yamnet


def yamnet_frames_tflite_model(feature_params):
    """Defines the YAMNet waveform-to-class-scores model,
    suitable for tflite conversion.

    Args:
      feature_params: An object with parameter fields to control the feature
      calculation.

    Returns:
      A model accepting (1, num_samples) waveform input and emitting a
      (num_patches, num_classes) matrix of class scores per time frame as
      well as a (num_spectrogram_frames, num_mel_bins) spectrogram feature
      matrix.
    """
    num_samples = int(round(params.SAMPLE_RATE * 0.975))
    waveform = layers.Input(batch_shape=(1, num_samples))
    # Store the intermediate spectrogram features to use in visualization.
    spectrogram = features_tflite_lib.waveform_to_log_mel_spectrogram(
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
    converter.experimental_new_converter = True
    tflite_model = converter.convert()
    open("yamnet.tflite", "wb").write(tflite_model)


if __name__ == '__main__':
    main()
