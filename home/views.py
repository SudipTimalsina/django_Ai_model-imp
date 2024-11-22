import numpy as np
import librosa
from django.conf import settings
from django.shortcuts import render
from django.core.files.storage import default_storage
from django.http import JsonResponse
import tensorflow as tf
from .forms import AudioUploadForm
from .models import birdlist

# Load the trained model
model = tf.keras.models.load_model(settings.BASE_DIR / 'my_model.keras', compile=False)

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

        mfccs = mfccs.T 

        # Reshape to match the model's input shape (1, 251, 21)
        audio_features = np.expand_dims(mfccs, axis=0)  

        prediction = model.predict(audio_features)
        predicted_class = np.argmax(prediction, axis=-1)[0]  

        # Get the class name from the mapping
        class_name = class_names.get(predicted_class, "Unknown")

        return class_name, prediction  # Return class name and probabilities

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
                    'probabilities': probabilities.tolist(),
                    'bird_details': bird_details
                })
            else:
                return JsonResponse({'error': 'Prediction failed'}, status=500)
    
    else:
        form = AudioUploadForm()
    
    return render(request, 'upload.html', {'form': form})
