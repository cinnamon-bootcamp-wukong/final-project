import cv2
import pandas as pd


class FaceDetector:
    """
    A class for detecting faces in images using a pre-trained Caffe model.

    Attributes:
        detector (cv2.dnn_Net): The face detection model loaded from Caffe.
        confidence_threshold (float): The threshold for considering a detected face as valid.
    """

    def __init__(self, prototxt_path, model_path, confidence_threshold=0.75):
        """
        Initializes the FaceDetector object with a specified Caffe model.

        Args:
            prototxt_path (str): Path to the .prototxt file that defines the model architecture.
            model_path (str): Path to the .caffemodel file that contains the pre-trained weights.
            confidence_threshold (float, optional): The minimum confidence level required to consider a detection as a face.
                                                    Defaults to 0.75.
        """
        self.detector = cv2.dnn.readNetFromCaffe(prototxt_path, model_path)
        self.confidence_threshold = confidence_threshold

    def detect_faces(self, img):
        """
        Detects faces in the given image.

        Args:
            img (bytes): The image data in bytes format, typically read from an image file.

        Returns:
            bool: True if a face is detected with a confidence level above the threshold, False otherwise.
        """
        image = cv2.imdecode(img, cv2.IMREAD_COLOR)
        target_size = (300, 300)
        image = cv2.resize(image, target_size)

        image_blob = cv2.dnn.blobFromImage(image=image)
        self.detector.setInput(image_blob)
        detections = self.detector.forward()

        column_labels = ["img_id", "is_face", "confidence", "left", "top", "right", "bottom"]
        df = pd.DataFrame(detections[0][0], columns=column_labels)

        is_face = df['is_face'][0]
        confidence = df['confidence'][0]

        return is_face == 1 and confidence > self.confidence_threshold
