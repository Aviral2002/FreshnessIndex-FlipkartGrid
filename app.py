import eventlet
eventlet.monkey_patch()

from flask import Flask
from flask_socketio import SocketIO, emit
from tensorflow.keras.models import load_model
import numpy as np
from PIL import Image
import io
import base64

# Initialize Flask and Flask-SocketIO
app = Flask(__name__)
socketio = SocketIO(app, cors_allowed_origins="*")

# Load the pre-trained CNN model
model = load_model('imageclassifier.h5')

# WebSocket route for continuous video frame classification
@socketio.on('message')
def handle_message(base64_image):
    try:
        # Decode the base64 string to get the image
        image_bytes = base64.b64decode(base64_image)
        img = Image.open(io.BytesIO(image_bytes))

        # Preprocess the image for the model
        img = img.resize((254, 254))  # Resize image to 254x254 pixels (as per model's input shape)
        img = np.array(img) / 255.0   # Normalize pixel values
        img = np.expand_dims(img, axis=0)  # Add batch dimension

        # Make a prediction using the model
        prediction = model.predict(img)

        # Determine classification based on the model's output
        if prediction[0][0] > 0.5:
            result = {'classification': 'rotten'}
        else:
            result = {'classification': 'fresh'}

        # Send the classification result back to the client
        emit('message', result)

    except Exception as e:
        # Handle any potential errors
        emit('error', {'error': str(e)})

# Run the Flask app with SocketIO
if __name__ == '__main__':
    socketio.run(app, host='0.0.0.0', port=5000)
