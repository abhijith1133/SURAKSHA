import cv2
import numpy as np
import time

# Load YOLOv5 model for object detection
def load_yolov5_model():
    from ultralytics import YOLO
    try:
        model = YOLO("yolov5s.pt")  # Pre-trained YOLOv5 model
        print("YOLOv5 model loaded successfully.")
        return model
    except Exception as e:
        print(f"Error loading YOLOv5 model: {str(e)}")
        exit()

# Detect objects in a frame
def detect_objects(model, frame):
    try:
        results = model(frame)
        detections = []
        for result in results:
            for box in result.boxes:
                class_id = int(box.cls)
                label = model.names[class_id]
                confidence = float(box.conf)
                bbox = box.xyxy[0].cpu().numpy()  # Bounding box coordinates
                detections.append({
                    "label": label,
                    "confidence": confidence,
                    "bbox": bbox
                })
        print("Detections:", detections)
        return detections
    except Exception as e:
        print(f"Error detecting objects: {str(e)}")
        return []

# Function to predict accident probability
def predict_accident_probability(vehicle_count, avg_speed, weather, has_emergency_vehicle, has_stray_animal):
    """
    Predict accident probability based on input features.
    This is a placeholder function for simulation purposes.
    """
    # Base probability based on vehicle count and speed
    accident_probability = (vehicle_count / 500) * (60 - avg_speed) / 60

    # Increase probability if there's a stray animal or emergency vehicle
    if has_stray_animal:
        accident_probability += 0.1  # +10% for stray animals
    if has_emergency_vehicle:
        accident_probability += 0.05  # +5% for emergency vehicles

    # Weather conditions (nighttime increases probability)
    if weather == "night":
        accident_probability *= 1.5  # Increase by 50% at night

    # Cap the probability between 0% and 100%
    accident_probability = min(max(accident_probability * 100, 0), 100)
    return round(accident_probability, 2)

# Process the camera feed
def process_camera_feed(video_path):
    """
    Process a video feed and simulate real-time predictions.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video at {video_path}")
        return

    # Reduce frame size and frame rate
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 10)

    model = load_yolov5_model()

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not read frame.")
            break

        # Resize the frame to a smaller size (e.g., 640x480)
        frame = cv2.resize(frame, (640, 480))

        # Detect objects in the frame
        detections = detect_objects(model, frame)

        # Analyze detections
        vehicle_count = sum(1 for det in detections if det["label"] in ["car", "bus"])
        has_emergency_vehicle = any(det["label"] == "ambulance" for det in detections)
        has_stray_animal = any(det["label"] in ["dog", "cat"] for det in detections)

        # Simulate average speed (random value for demonstration)
        avg_speed = np.random.randint(10, 60)

        # Simulate weather condition
        weather = "night"  # Fixed for nighttime simulation

        # Predict accident probability
        accident_probability = predict_accident_probability(vehicle_count, avg_speed, weather, has_emergency_vehicle, has_stray_animal)
        print(f"Vehicle Count: {vehicle_count}, Avg Speed: {avg_speed}, Accident Probability: {accident_probability}%")

        # Overlay predictions on the video frame
        cv2.putText(
            frame,
            f"Vehicle Count: {vehicle_count}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2
        )
        cv2.putText(
            frame,
            f"Avg Speed: {avg_speed} km/h",
            (10, 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2
        )
        cv2.putText(
            frame,
            f"Weather: {weather}",
            (10, 90),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2
        )
        cv2.putText(
            frame,
            f"Emergency Vehicle: {'Yes' if has_emergency_vehicle else 'No'}",
            (10, 120),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 0, 255),
            2
        )
        cv2.putText(
            frame,
            f"Stray Animal: {'Yes' if has_stray_animal else 'No'}",
            (10, 150),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 0, 255),
            2
        )
        cv2.putText(
            frame,
            f"Accident Prob: {accident_probability}%",
            (10, 180),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 0, 255),
            2
        )

        # Display the frame with predictions
        cv2.imshow("Real-Time Traffic Monitoring", frame)

        # Exit the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        # Simulate real-time delay (e.g., 1 second between frames)
        time.sleep(1)

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    # Path to the video file (simulating a camera feed)
    VIDEO_PATH = "./data/congestion.mp4"

    # Process the camera feed
    process_camera_feed(VIDEO_PATH)