from pushbullet import PushBullet
import cv2
import os
import time
from deepface import DeepFace


API_KEY = "o.IOTNvyg609CJAjdRbQrxGfgUELtpM495" #-> Fake API KEY

file = 'motion.txt'
with open(file, 'r') as file:
    message = file.readlines()
    message = ''.join(message)



def capture_and_verify(verification_image_path):
    # Open the video source (webcam)
    vid = cv2.VideoCapture(0)

    try:
        while True:
            # Capture a frame from the webcam
            ret, frame = vid.read()
            if ret:
                capture_image_path = 'current_frame.jpg'
                # Save the captured frame to a file
                cv2.imwrite(capture_image_path, frame)

                try:
                    # Verify the captured image against the verification image using DeepFace
                    result = DeepFace.verify(verification_image_path, capture_image_path)
                    if result['verified']:
                        print("Face verified!")
                        # Perform actions on successful verification
                    else:
                        print("Face not verified!")
                        pb = PushBullet(API_KEY)
                        push = pb.push_note("Warning!",message)
                except Exception as e:
                    print("Verification error:", e)
                finally:
                    # Remove the captured image file
                    if os.path.exists(capture_image_path):
                        os.remove(capture_image_path)
            else:
                print("Failed to capture image")

            # Display the frame
            cv2.imshow('Video', frame)

            # Check for 'q' key to exit
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

            # Wait for 2 seconds before capturing the next frame
            time.sleep(2)

    finally:
        # Release the video source and close all windows
        vid.release()
        cv2.destroyAllWindows()

# Path to the image used for verification
verification_image_path = 'My_Face.jpg'


# Capture and verify the face every 2 seconds
capture_and_verify(verification_image_path)
