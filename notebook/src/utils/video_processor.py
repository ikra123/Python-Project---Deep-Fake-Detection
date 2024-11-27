import cv2
import numpy as np

class VideoProcessor:
    def __init__(self, source=0):
        self.video_capture = cv2.VideoCapture(source)

    def capture_frame(self):
        ret, frame = self.video_capture.read()
        return ret, frame

    def resize_frame(self, frame, scale=0.25):
        return cv2.resize(frame, (0, 0), fx=scale, fy=scale)
    
    def normalize_frame(self , frame , face_location , resize):
        return cv2.resize(
            frame[
                face_location[0]:face_location[2],
                face_location[3]:face_location[1]
            ], resize) / 255.0
    
    def reshape_frame(self , face_image , reshape):
        return np.reshape(face_image, reshape)

    def convert_color(self, frame):
        return frame[:, :, ::-1]

    def display_results(self, frame, face_locations, face_detections):
        for (top, right, bottom, left), face_detection in zip(face_locations, face_detections):
            top *= 4
            right *= 4
            bottom *= 4
            left *= 4

            label = "Fake" if face_detection > 0.5 else "Real"
            cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
            font = cv2.FONT_HERSHEY_DUPLEX
            cv2.putText(frame, label, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)

        cv2.imshow('Video', frame)
        
    def quit(self):
        return cv2.waitKey(1) & 0xFF == ord('q')
    
    def release(self):
        self.video_capture.release()
        cv2.destroyAllWindows()