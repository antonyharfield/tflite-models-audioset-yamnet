
1. Download the model weights

   ```
   curl ...
   ```

2. Open a TF environment

   ```
   docker run -it --rm -v ${PWD}:/app -w /app tensorflow/tensorflow:2.2.0rc0-py3
   docker run -it --rm -v ${PWD}:/app -w /app tensorflow/tensorflow:2.1.0-py3
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

5. 

   ```
   python inference_tflite.py dog_112_975ms.wav
   ```