import cv2
import pandas as pd

class FaceDetector:
    def __init__(self, prototxt_path, model_path, confidence_threshold=0.75):
        self.detector = cv2.dnn.readNetFromCaffe(prototxt_path, model_path)
        self.confidence_threshold = confidence_threshold

    def detect_faces(self, img):
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