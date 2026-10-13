import cv2
import numpy as np

FLOW_WIDTH = 320

class MotionEstimator:
    def __init__(self):
        self.prev_gray = None
        self.scale = None

    def compute_flow(self, frame):
        h, w = frame.shape[:2]
        scale = FLOW_WIDTH / w
        small = cv2.resize(frame, (FLOW_WIDTH, int(h * scale)))
        gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
        if self.prev_gray is None:
            self.prev_gray = gray
            self.scale = scale
            return None
        flow = cv2.calcOpticalFlowFarneback(
            self.prev_gray, gray, None,
            pyr_scale=0.5, levels=2, winsize=13,
            iterations=2, poly_n=5, poly_sigma=1.1, flags=0
        )
        self.prev_gray = gray
        return flow

    def person_vector(self, flow, bbox):
        x1, y1, x2, y2 = [int(v * self.scale) for v in bbox]
        region = flow[y1:y2, x1:x2]
        if region.size == 0:
            return (0.0, 0.0)
        dx = float(np.mean(region[..., 0])) / self.scale
        dy = float(np.mean(region[..., 1])) / self.scale
        return (dx, dy)