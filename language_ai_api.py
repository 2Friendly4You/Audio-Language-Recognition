import flask
from flask import request, jsonify
import flask_cors
import tensorflow as tf
import librosa
import json

app = flask.Flask(__name__)
flask_cors.CORS(app)
model = None
target_shape = (128, 128)
classes = ['de', 'en', 'es', 'fr', 'it', 'pt', 'ru', 'tr']

# a audio file is sent to the server and the server returns the ai classification
@app.route('/api/v1/language_classification', methods=['POST'])
def ai():
    # Get the audio file from the request
    audio_file = request.files['audio']
    audio_file.save('audio_file.wav')
    
    # Get the class probabilities and predicted class index
    class_probabilities, predicted_class_index = test_audio('audio_file.wav', model)

    # Get the predicted class
    predicted_class = classes[predicted_class_index]

    # Return the predicted class and class probabilities
    return jsonify({'predicted_class': predicted_class, 'class_probabilities': class_probabilities.tolist()})
    
# Function to preprocess and classify an audio file
def test_audio(file_path, model):
    # Load and preprocess the audio file
    audio_data, sample_rate = librosa.load(file_path, sr=None)
    mel_spectrogram = librosa.feature.melspectrogram(y=audio_data, sr=sample_rate)
    mel_spectrogram = tf.image.resize(np.expand_dims(mel_spectrogram, axis=-1), target_shape)
    mel_spectrogram = tf.reshape(mel_spectrogram, (1,) + target_shape + (1,))
    
    # Make predictions
    predictions = model.predict(mel_spectrogram)
    
    # Get the class probabilities
    class_probabilities = predictions[0]
    
    # Get the predicted class index
    predicted_class_index = np.argmax(class_probabilities)
    
    return class_probabilities, predicted_class_index

if __name__ == '__main__':
    # load the model 
    model = tf.keras.models.load_model('audio_classification_model.keras')
    app.run(port=5000)