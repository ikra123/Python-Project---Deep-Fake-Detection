from utils.deepfake_detector import DeepFakeDetector
from utils.face_recognizer import FaceRecognizer
from utils.video_processor import VideoProcessor

def main():
    # Create instances of the classes
    deepfake_detector = DeepFakeDetector('models/xception_deepfake_image.h5')
    face_recognizer = FaceRecognizer()
    video_processor = VideoProcessor(0)  # Webcam #0


    while True:
        ret , frame = video_processor.capture_frame()

        small_frame = video_processor.resize_frame(frame)
        rgb_small_frame = video_processor.convert_color(small_frame)

        face_locations = face_recognizer.detect_faces(rgb_small_frame)
        face_encodings = face_recognizer.encode_faces(rgb_small_frame , face_locations)
        faces_detection = []
        
        for face_location, face_encoding in zip(face_locations, face_encodings):
            face_image = video_processor.normalize_frame(frame, face_location, (224, 224))
            face_image = video_processor.reshape_frame(face_image, (1, 224, 224, 3))

            prediction = deepfake_detector.predict(face_image)
            faces_detection.append(prediction)


        video_processor.display_results(frame, face_locations, faces_detection)

        if video_processor.quit():
            break

    video_processor.release()

if __name__ == "__main__":
    main()