import numpy as np
import tensorflow as tf

class RfcxFrame(tf.Module):
    def __init__(self, model, sample_rate, audio_length_secs, class_names, codec):
        self.model = model
        self._sample_rate = sample_rate
        self._context_width_samples = int(sample_rate * audio_length_secs)
        self._class_names = class_names
        self._codec = codec

    @tf.function(
        input_signature=[
            tf.TensorSpec(shape=(None, None, 1), dtype=tf.float32),
            tf.TensorSpec(shape=tuple(), dtype=tf.int64),
        ]
    )
    def score(self, waveform, context_step_samples):
        scores, _ = self.model(tf.squeeze(waveform, 2)) # remove channel
        scores = tf.expand_dims(scores, 0) # add batch

        return {"scores": scores}

    @tf.function(input_signature=[])
    def metadata(self):
        return {
            "input_sample_rate": self._sample_rate,
            "context_width_samples": self._context_width_samples,
            "class_names": np.array(self._class_names),
            "codec": self._codec
        }