import face_recognition

class FaceRecognizer:
    # detecte face for getting the faces location
    def detect_faces(self, frame):
        return face_recognition.face_locations(frame)
    
    def encode_faces(self , frame , face_locations):
        return face_recognition.face_encodings(frame, face_locations)