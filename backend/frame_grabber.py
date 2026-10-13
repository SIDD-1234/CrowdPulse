import cv2
import threading
import time

class LatestFrameReader:
    def __init__(self, source, max_retries=3):
        self.lock = threading.Lock()
        self.frame = None
        self.ret = False
        self.stopped = False
        self.cap = None

        for attempt in range(max_retries):
            self.cap = cv2.VideoCapture(source)
            if self.cap.isOpened():
                break
            time.sleep(0.5)

        if self.cap is not None and self.cap.isOpened():
            self.thread = threading.Thread(target=self._update, daemon=True)
            self.thread.start()

    def isOpened(self):
        return self.cap is not None and self.cap.isOpened()

    def _update(self):
        while not self.stopped:
            try:
                ret, frame = self.cap.read()
            except Exception:
                ret, frame = False, None
            with self.lock:
                self.ret, self.frame = ret, frame
            if not ret:
                break

    def read(self):
        with self.lock:
            return self.ret, (self.frame.copy() if self.frame is not None else None)

    def release(self):
        self.stopped = True
        if hasattr(self, "thread") and self.thread.is_alive():
            self.thread.join(timeout=2)
        if self.cap is not None:
            self.cap.release()