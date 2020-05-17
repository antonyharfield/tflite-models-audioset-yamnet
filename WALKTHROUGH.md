
1. Download the model weights

   ```
   curl -o yamnet.h5 https://storage.googleapis.com/audioset/yamnet.h5
   ```

2. Open a TF environment (tested on TF 2.1 and 2.2)

   ```
   docker run -it --rm -v ${PWD}:/app -w /app tensorflow/tensorflow:2.2.0
   ```

3. Install the dependencies

   ```
   apt-get install -y libsndfile1
   pip install resampy soundfile matplotlib
   ```

4. Generate the tflite model from the original YAMNet

   ```
   python to_tflite.py
   ```

5. Run an inference to test the tflite model

   ```
   python inference_tflite.py dog_112_975ms.wav
   ```