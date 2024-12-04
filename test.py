import cv2
import numpy as np
from insightface.app import FaceAnalysis
import matplotlib.pyplot as plt
import pdb
import time

# Initialize the FaceAnalysis application
app = FaceAnalysis(name='buffalo_l')
app.prepare(ctx_id=0)

# Replace with your RTSP stream URL
RTSP_URL = "rtsp://duodecimal:anandji99@169.254.105.8:554/Streaming/Channels/101"

def main():
    # Open the RTSP stream
    cap = cv2.VideoCapture(RTSP_URL)
    if not cap.isOpened():
        print("Error: Could not open video stream.")
        return

    # Frame rate control variables
    desired_fps = 0.5  # Adjust this to your desired frame rate
    frame_delay = 1 / desired_fps  # Calculate delay between frames

    while True:
        try:
            start_time = time.time()

            success, frame = cap.read()
            if not success:
                print("Can't receive frame. Retrying ...")
                cap.release()
                cap = cv2.VideoCapture(RTSP_URL)
                success, frame = cap.read()

            # Perform face analysis on the captured frame
            faces = app.get(frame)

            # Annotate detected faces in the frame
            annotated_img = frame.copy()
            for face in faces:
                bbox = face.bbox.astype(int)
                cv2.rectangle(annotated_img, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 255, 0), 2)

            # Display the annotated image
            cv2.imshow("Real-Time Face Detection", annotated_img)

            # Control frame rate
            elapsed_time = time.time() - start_time
            if elapsed_time < frame_delay:
                time.sleep(frame_delay - elapsed_time)

            # Exit on 'q' key press
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        except Exception as e:
            pdb.set_trace()
            print("EXCEPTION:", e)

    # Release resources
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()