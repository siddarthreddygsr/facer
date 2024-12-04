import cv2
import numpy as np
from insightface.app import FaceAnalysis
import threading
import queue
import time
import json

class VideoStreamReader:
    def __init__(self, rtsp_url):
        self.frame_queue = queue.Queue(maxsize=128)
        self.rtsp_url = rtsp_url
        self.stop_event = threading.Event()
        self.thread = threading.Thread(target=self.read_frames, daemon=True)

    def start(self):
        self.thread.start()

    def read_frames(self):
        cap = cv2.VideoCapture(self.rtsp_url)
        
        if not cap.isOpened():
            print("Error: Could not open video stream.")
            self.stop_event.set()
            return

        while not self.stop_event.is_set():
            try:
                success, frame = cap.read()
                if not success:
                    print("Can't receive frame. Retrying...")
                    cap.release()
                    cap = cv2.VideoCapture(self.rtsp_url)
                    continue

                # If queue is full, remove the oldest frame
                if not self.frame_queue.empty() and self.frame_queue.full():
                    try:
                        self.frame_queue.get_nowait()
                    except queue.Empty:
                        pass

                # Put the new frame in the queue
                self.frame_queue.put(frame, block=False)

            except Exception as e:
                print(f"Frame reading error: {e}")
                break

        cap.release()

    def get_frame(self):
        try:
            return self.frame_queue.get(timeout=2)
        except queue.Empty:
            return None

    def stop(self):
        self.stop_event.set()
        self.thread.join()

class FaceDetector:
    def __init__(self):
        # Initialize the FaceAnalysis application
        self.app = FaceAnalysis(name='buffalo_s')
        self.app.prepare(ctx_id=0)
        self.loaded_embeddings = self.load_face_embeddings("face_embeddings.json")
    
    def load_face_embeddings(self, embeddings_file):
        with open(embeddings_file, 'r') as f:
            face_embeddings = json.load(f)
        return face_embeddings
    

    def compare_face_embeddings(self, input_embedding, stored_embeddings, threshold=0.5):
        from numpy.linalg import norm
        
        matches = []
        
        for person_name, person_embeddings in stored_embeddings.items():
            for stored_embedding in person_embeddings:
                # Calculate cosine similarity
                cos_sim = np.dot(input_embedding, stored_embedding) / (norm(input_embedding) * norm(stored_embedding))
                
                matches.append({'name':person_name, 'score':round(float(cos_sim), 2)})
                # if cos_sim > threshold:
                    # break
        
        return matches

    def detect_faces(self, frame):
        # Perform face analysis on the captured frame
        faces = self.app.get(frame)
        
        # Annotate detected faces in the frame
        annotated_img = frame.copy()
        for face in faces:
            bbox = face.bbox.astype(int)
            name = self.compare_face_embeddings(face.normed_embedding, self.loaded_embeddings, 0.5)
            if len(name) > 0:
                name = name[0]
            else:
                name = None
            name = str(name)
            cv2.rectangle(annotated_img, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 255, 0), 2)
            text_position = (bbox[0], bbox[1] - 10)
            cv2.putText(annotated_img, name, text_position, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        
        return annotated_img

def main():
    # Replace with your RTSP stream URL
    RTSP_URL = "rtsp://duodecimal:anandji99@169.254.105.8:554/Streaming/Channels/101"

    # Initialize components
    stream_reader = VideoStreamReader(RTSP_URL)
    face_detector = FaceDetector()

    # Start frame reading thread
    stream_reader.start()

    # Frame rate control variables
    desired_fps = 15  # Adjust this to your desired frame rate
    frame_delay = 1 / desired_fps  # Calculate delay between frames

    try:
        while True:
            start_time = time.time()

            # Get the latest frame
            frame = stream_reader.get_frame()
            
            if frame is not None:
                # Detect and annotate faces
                annotated_img = face_detector.detect_faces(frame)

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
        print(f"Main loop error: {e}")

    finally:
        # Clean up
        stream_reader.stop()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()