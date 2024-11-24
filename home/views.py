import numpy as np
import librosa
from django.conf import settings
from django.shortcuts import render
from django.core.files.storage import default_storage
from django.http import JsonResponse
import tensorflow as tf
from .forms import AudioUploadForm
from .models import birdlist

# If model loading raises errors, try reconstructing the model
def build_model():
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Input(shape=(251, 21)))  # Replace batch_input_shape with Input layer

    # Recreate GRU layer without the 'batch_input_shape' and 'time_major' arguments
    model.add(tf.keras.layers.GRU(64, return_sequences=True))  # Modify units as per your model

    # Add other layers accordingly based on your model architecture
    model.add(tf.keras.layers.Dense(3, activation='softmax'))  # Example final layer

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    return model

# Global variable to hold the model
model = None

def load_model():
    global model
    try:
        # Try loading the existing model
        model = tf.keras.models.load_model(settings.BASE_DIR / 'my_model1.keras')
        print("Model loaded successfully")
    except Exception as e:
        print(f"Model loading failed: {e}")
        # If loading fails, try building a new model (if possible)
        model = build_model()

# Load model initially
load_model()

# Class Mapping
class_names = {0: 'Scarlet-chested Sunbird', 1: 'Egyptian Goose', 2: 'Woodland Kingfisher'}

def predict_audio(file_path):
    try:
        # Load the audio file
        audio, sr = librosa.load(file_path, sr=None)

        # Extract MFCCs (21 MFCCs to match model input)
        mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=21)

        # Ensure the time steps (frames) match the model's expected input (251 time steps in this case)
        required_time_steps = 251

        if mfccs.shape[1] < required_time_steps:
            # Pad with zeros if shorter
            pad_width = required_time_steps - mfccs.shape[1]
            mfccs = np.pad(mfccs, pad_width=((0, 0), (0, pad_width)), mode='constant')
        else:
            # Truncate if longer
            mfccs = mfccs[:, :required_time_steps]

        # Transpose to match expected input shape (251 time steps, 21 features)
        mfccs = mfccs.T  

        # Reshape to match the model's input shape (1, 251, 21)
        audio_features = np.expand_dims(mfccs, axis=0)  

        # Make the prediction
        prediction = model.predict(audio_features)

        # Get the prediction from the last time step (index 250)
        last_time_step_prediction = prediction[0, -1, :]  # (3,)

        # Apply np.argmax to get the class with the highest probability at the last time step
        predicted_class = np.argmax(last_time_step_prediction)

        # Get the class name from the mapping
        class_name = class_names.get(predicted_class, "Unknown")

        return class_name, last_time_step_prediction.tolist()  # Return class name and the probabilities

    except Exception as e:
        print(f"Error during prediction: {e}")
        return None, None

def upload_and_predict(request):
    if request.method == 'POST':
        form = AudioUploadForm(request.POST, request.FILES)
        if form.is_valid():
            # Save the uploaded file temporarily
            audio_file = request.FILES['audio_file']
            file_path = default_storage.save(f'temp/{audio_file.name}', audio_file)
            file_path_full = default_storage.path(file_path)

            # Make prediction
            class_name, probabilities = predict_audio(file_path_full)

            # Clean up the temporary file
            default_storage.delete(file_path)

            if class_name:
                # Fetch additional bird information from the database
                bird_info = birdlist.objects.filter(name__iexact=class_name).first()

                if bird_info:
                    bird_details = {
                        'ScientificName': bird_info.scientificName,
                        'MoreInfo': bird_info.birdUrl,
                    }
                else:
                    bird_details = {}

                # Return the JSON response with the bird class, probabilities, and details
                return JsonResponse({
                    'class_name': class_name,
                    'probabilities': probabilities,
                    'bird_details': bird_details
                })
            else:
                return JsonResponse({'error': 'Prediction failed'}, status=500)

    else:
        form = AudioUploadForm()

    return render(request, 'upload.html', {'form': form})
