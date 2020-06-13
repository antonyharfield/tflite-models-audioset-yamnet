import tensorflow as tf
import params
import yamnet


def main():
    # Load the model and weights
    model = yamnet.yamnet_frames_model(params)
    model.load_weights('yamnet.h5')

    # Convert the model
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    tflite_model = converter.convert()
    open("yamnet.tflite", "wb").write(tflite_model)


if __name__ == '__main__':
    main()
