import cv2
import numpy as np
from streamer import VideoStreamReader
# from streamer2 import VideoStreamReader
import time
from processor import FaceDetector
from embeddings import create_face_embeddings




def main():
    RTSP_URL = "rtsp://duodecimal:anandji99@169.254.105.8:554/Streaming/Channels/101"
    embedding_file = 'face_embeddings.json'
    model_name = 'buffalo_l'
    create_face_embeddings('../data', embedding_file, model_name)
    # stream_reader = VideoStreamReader()
    stream_reader = VideoStreamReader(RTSP_URL)
    face_detector = FaceDetector(model_name)

    stream_reader.start()

    desired_fps = 30
    frame_delay = 1 / desired_fps

    try:
        while True:
            start_time = time.time()
            frame = stream_reader.get_frame()
            
            if frame is not None:
                annotated_img = face_detector.detect_faces(frame)
                cv2.imshow("Real-Time Face Detection", annotated_img)
            elapsed_time = time.time() - start_time
            if elapsed_time < frame_delay:
                time.sleep(frame_delay - elapsed_time)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    except Exception as e:
        print(f"Main loop error: {e}")

    finally:
        stream_reader.stop()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()