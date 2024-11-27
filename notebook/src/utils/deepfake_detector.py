from keras.models import load_model
import numpy as np

class DeepFakeDetector:
    def __init__(self, model_path):
        self.model = load_model(model_path)

    def predict(self, face_image):
        # Make prediction using the deep fake detection model
        prediction = self.model.predict(face_image)
        return prediction[0][0] 