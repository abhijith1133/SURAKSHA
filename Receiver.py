
import firebase_admin
from firebase_admin import credentials, db

# Initialize Firebase
cred = credentials.Certificate(r"C:\Users\User\Downloads\suraksha-d204f-firebase-adminsdk-fbsvc-df7f3afd86.json")
firebase_admin.initialize_app(cred, {
    'databaseURL': 'https://suraksha-d204f-default-rtdb.asia-southeast1.firebasedatabase.app'
})

# Function to listen for real-time alerts
def listen_for_hazards():
    hazard_ref = db.reference('hazards')
    def callback(event):
        print(f"New Hazard Detected: {event.data}")
    hazard_ref.listen(callback)

if __name__ == "__main__":
    listen_for_hazards()