import threading
import queue
import cv2

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
                if not self.frame_queue.empty() and self.frame_queue.full():
                    try:
                        self.frame_queue.get_nowait()
                    except queue.Empty:
                        pass
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