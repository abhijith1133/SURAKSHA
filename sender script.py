import cv2
import torch
import numpy as np
import firebase_admin
from firebase_admin import credentials, db

# ===================== Firebase Setup =====================
cred = credentials.Certificate(r"C:\Users\anshi\OneDrive\Desktop\suraksha-d204f-firebase-adminsdk-fbsvc-df7f3afd86.json")
firebase_admin.initialize_app(cred, {
    'databaseURL': 'https://suraksha-d204f-default-rtdb.asia-southeast1.firebasedatabase.app'
})

# Function to send hazard alerts to Firebase
def send_hazard_alert(hazard_type, location):
    hazard_ref = db.reference('hazards')
    hazard_ref.push({
        "type": hazard_type,
        "location": location,
        "timestamp": {".sv": "timestamp"}  # Automatically add timestamp
    })

# ===================== YOLOv5 Setup =====================
model = torch.hub.load('ultralytics/yolov5', 'yolov5s')  # Pre-trained YOLOv5 model
CLASSES = model.names
# Define classes of interest
EMERGENCY_VEHICLES = ['ambulance', 'fire truck', 'police car']
ANIMALS = ['dog', 'cat', 'cow', 'horse']  # Add more animals if needed
TRAFFIC_LIGHTS = ['traffic light']  # Traffic light class
TRAFFIC_SIGNS = ['stop sign', 'speed limit sign']  # Add more traffic signs if needed

# ===================== Lane Detection =====================
def detect_lane_deviation(frame):
    # Convert frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # Apply edge detection
    edges = cv2.Canny(gray, 50, 150)
    # Define region of interest (ROI) for lanes
    height, width = frame.shape[:2]
    mask = np.zeros_like(edges)
    polygon = np.array([[(0, height), (width // 2, height // 2), (width, height)]], dtype=np.int32)
    cv2.fillPoly(mask, polygon, 255)
    masked_edges = cv2.bitwise_and(edges, mask)
    # Detect lines using Hough Transform
    lines = cv2.HoughLinesP(masked_edges, 1, np.pi / 180, threshold=50, minLineLength=50, maxLineGap=100)
    if lines is None:
        return "Lane deviation detected!", True
    return "Lane OK", False

# ===================== Video Processing =====================
def process_video(video_path):
    cap = cv2.VideoCapture(video_path)
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Resize frame for faster processing
        frame = cv2.resize(frame, (640, 480))

        # Lane deviation detection
        lane_status, deviation = detect_lane_deviation(frame)
        if deviation:
            cv2.putText(frame, "WARNING: Lane Deviation!", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            location = "10.0123,76.1234"  # Replace with actual GPS coordinates
            send_hazard_alert("Lane Deviation", location)

        # Object detection with YOLOv5
        results = model(frame)
        detections = results.pandas().xyxy[0]  # Get detection results
        for _, detection in detections.iterrows():
            label = detection['name']
            confidence = detection['confidence']
            x1, y1, x2, y2 = int(detection['xmin']), int(detection['ymin']), int(detection['xmax']), int(detection['ymax'])

            # Draw bounding box
            color = (0, 255, 0)  # Default color
            alert_message = ""

            # Stray animal alert
            if label in ANIMALS:
                color = (0, 0, 255)  # Red
                alert_message = f"WARNING: Stray {label} ahead!"
                location = "10.0123,76.1234"  # Replace with actual GPS coordinates
                send_hazard_alert(f"Stray {label}", location)

            # Emergency vehicle alert
            elif label in EMERGENCY_VEHICLES:
                color = (255, 0, 0)  # Blue
                alert_message = f"WARNING: {label.capitalize()} approaching!"
                location = "10.0123,76.1234"  # Replace with actual GPS coordinates
                send_hazard_alert(f"{label.capitalize()} Approaching", location)

            # Traffic light detection
            elif label in TRAFFIC_LIGHTS:
                color = (0, 255, 255)  # Yellow
                alert_message = "Traffic Light Detected"
                # Simulate traffic light state (you can classify red/green using additional logic)
                location = "10.0123,76.1234"  # Replace with actual GPS coordinates
                send_hazard_alert("Traffic Light", location)

            # Traffic sign detection
            elif label in TRAFFIC_SIGNS:
                color = (255, 255, 0)  # Cyan
                alert_message = f"Traffic Sign Detected: {label}"
                location = "10.0123,76.1234"  # Replace with actual GPS coordinates
                send_hazard_alert(f"Traffic Sign: {label}", location)

            # Congestion alert (based on vehicle density)
            vehicle_count = len(detections[detections['name'].isin(['car', 'truck', 'bus'])])
            if vehicle_count > 3:  # Arbitrary threshold for congestion
                alert_message = "WARNING: Traffic congestion ahead!"
                location = "10.0123,76.1234"  # Replace with actual GPS coordinates
                send_hazard_alert("Traffic Congestion", location)

            # Display alerts
            if alert_message:
                cv2.putText(frame, alert_message, (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

            # Draw bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, f"{label} {confidence:.2f}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

        # Show the frame
        cv2.imshow("Smart Kerala Highway Companion", frame)

        # Exit on 'q' key press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

# ===================== Main Execution =====================
if __name__ == "__main__":
    video_path = r"C:\Users\anshi\OneDrive\Desktop\kpv.MOV"  # Replace with your video file path
    process_video(video_path)