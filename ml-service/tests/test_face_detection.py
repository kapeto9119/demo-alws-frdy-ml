import sys
import os
import pytest

# Append the project root directory to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from app.services.face_detection import detect_faces_in_image

def test_detect_faces_in_image_with_valid_input():
    with open("tests/test_image.jpg", "rb") as f:
        image_bytes = f.read()
    result = detect_faces_in_image(image_bytes, 1.1, 4)
    assert result is not None


def test_detect_faces_in_image_with_invalid_input():
    image_bytes = b"not an image"
    result = detect_faces_in_image(image_bytes, 1.1, 4)
    assert result is None

def test_detect_faces_in_image_with_invalid_parameters():
    with open("tests/test_image.jpg", "rb") as f:
        image_bytes = f.read()
    result = detect_faces_in_image(image_bytes, 0.5, -1)
    assert result is None