import flask
from flask import request, jsonify
import flask_cors
import tensorflow as tf
import json
import numpy as np

app = flask.Flask(__name__)
flask_cors.CORS(app)
model = None
target_shape = (256, 256)
classes = ['de', 'en', 'es', 'fr', 'it', 'pt', 'ru', 'tr']

# a audio file is sent to the server and the server returns the ai classification


@app.route('/api/v1/language_classification', methods=['POST'])
def ai():
    try:
        # Get the audio file from the request
        audio_file = request.files['audio']
        audio_file.save('audio.wav')

        # Get the class probabilities and predicted class index
        class_probabilities, predicted_class_index = test_audio('audio.wav', model)

        # Get the predicted class
        predicted_class = classes[predicted_class_index]

        # send a sorted list of classes and their probabilities
        class_probabilities = class_probabilities.tolist()
        class_probabilities = [{'class': classes[i], 'probability': class_probabilities[i]} for i in range(len(classes))]
        class_probabilities = sorted(class_probabilities, key=lambda x: x['probability'], reverse=True)

        # Return the predicted class and class probabilities
        return jsonify({'predicted_class': predicted_class, 'class_probabilities': class_probabilities}), 200
    except Exception as e:
        print('Error:', str(e))
        return jsonify({'error': str(e)}), 500


# Function to preprocess and classify an audio file
def test_audio(file_path, model):
    # Load and preprocess the audio file
    audio_binary = tf.io.read_file(file_path)
    audio, sample_rate = tf.audio.decode_wav(audio_binary)
    waveform = tf.squeeze(audio, axis=-1)
    mel_spectrogram = get_spectrogram(waveform)
    mel_spectrogram = tf.image.resize(mel_spectrogram, target_shape)
    mel_spectrogram = mel_spectrogram.numpy()
    mel_spectrogram = mel_spectrogram[np.newaxis, ...]
    
    # Make predictions
    predictions = model.predict(mel_spectrogram)
    
    # Get the class probabilities
    class_probabilities = predictions[0]
    
    # Get the predicted class index
    predicted_class_index = np.argmax(class_probabilities)
    
    return class_probabilities, predicted_class_index


# Function to convert waveform to spectrogram
def get_spectrogram(waveform):
    # Convert the waveform to a spectrogram using Short-Time Fourier Transform (STFT)
    spectrogram = tf.signal.stft(waveform, frame_length=255, frame_step=128)
    # Obtain the magnitude of the STFT
    spectrogram = tf.abs(spectrogram)
    # Add a `channels` dimension so that the spectrogram can be used as image-like input data
    spectrogram = spectrogram[..., tf.newaxis]
    return spectrogram


if __name__ == '__main__':
    # load the model
    model = tf.keras.models.load_model('latest_model.keras')
    app.run(port=5000, debug=True)
