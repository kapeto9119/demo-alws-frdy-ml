import cv2
import numpy as np
from app.utils.logger import logger

# Load the pre-trained face detection model (Haar Cascade)
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

def detect_faces_in_image(image_bytes: bytes, scale_factor: float, min_neighbors: int):
    try:
        nparr = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if img is None:
            logger.error("Image decoding failed")
            return None

        # Resize image if too large
        max_dimension = 800
        height, width = img.shape[:2]
        if max(height, width) > max_dimension:
            scaling_factor = max_dimension / float(max(height, width))
            img = cv2.resize(
                img,
                None,
                fx=scaling_factor,
                fy=scaling_factor,
                interpolation=cv2.INTER_AREA,
            )

        # Convert to grayscale for detection
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Detect faces
        faces = face_cascade.detectMultiScale(
            gray, scaleFactor=scale_factor, minNeighbors=min_neighbors
        )

        # Draw rectangles around faces
        for (x, y, w, h) in faces:
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # Encode image as JPEG
        success, img_encoded = cv2.imencode(".jpg", img)
        if not success:
            logger.error("Image encoding failed")
            return None

        return img_encoded.tobytes()
    except Exception as e:
        logger.exception("Error in face detection")
        return None