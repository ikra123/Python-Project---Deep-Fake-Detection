# DeepFake Detection App

This is a Python application that uses a deep learning model to detect deepfakes in real-time video. The application captures video from the webcam, detects faces in each frame, and uses the deepfake detection model to predict whether each face is real or fake.

## Installation

1. Clone this repository to your local machine.
2. Navigate to the project directory: `cd my-deepfake-detection-app`
3. Install the required Python packages: `pip install -r requirements.txt`

## Usage

1. Navigate to the `src` directory: `cd src`
2. Run the main script: `python main.py`
3. The application will start capturing video from your webcam. For each detected face, it will display a label indicating whether the face is real or fake.
4. To quit the application, press 'q' on your keyboard.

## Dependencies

This application requires the following Python packages:

- face_recognition
- cv2
- numpy
- keras

These dependencies can be installed using the `requirements.txt` file.

## Contributing

Contributions are welcome. Please submit a pull request with your proposed changes.

## License

This project is licensed under the MIT License. See the `LICENSE` file for more details.