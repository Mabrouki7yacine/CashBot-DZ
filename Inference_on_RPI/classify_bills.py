import cv2
import numpy as np
from picamera2 import Picamera2

# Config
MODEL_PATH = "update/path/to/trained_model.onnx"
CLASS_LABELS = ["1000", "2000", "unknown"]
INPUT_SIZE = (224, 224)

# Start OpenCV thread for display
cv2.startWindowThread()

# Initialize Picamera2
picam2 = Picamera2()
picam2.configure(picam2.create_preview_configuration(main={"format": 'XRGB8888'}))
picam2.start()

# Image preprocessing
def preprocess(image):
    img = cv2.resize(image, INPUT_SIZE)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    img = img.astype(np.float32) / 255.0
    img = (img - 0.5) / 0.5
    return cv2.dnn.blobFromImage(img, size=INPUT_SIZE)

# Predict and annotate
def predict_and_display(frame, net):
    blob = preprocess(frame)
    net.setInput(blob)
    output = net.forward()
    class_id = int(np.argmax(output))
    confidence = float(output[0][class_id])

    label = CLASS_LABELS[class_id] if 0 <= class_id < len(CLASS_LABELS) else "Unknown"
    text = f"{label} ({confidence*100:.1f}%)"

    cv2.putText(frame, text, (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

# Main
def main():
    print("Loading model...")
    net = cv2.dnn.readNetFromONNX(MODEL_PATH)
    if net.empty():
        print("Failed to load the ONNX model.")
        return

    print("Webcam started. Press 'q' to quit.")
    while True:
        image = picam2.capture_array()
        predict_and_display(image, net)
        cv2.imshow("Live Classification", image)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
