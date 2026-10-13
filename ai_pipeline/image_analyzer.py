from density_model import DensityEstimator
from config import settings
import cv2
import numpy as np
import base64
import tempfile
import os

class ImageAnalyzer:
    def __init__(self):
        self.density_estimator = DensityEstimator()

    def analyze(self, frame):
        h, w = frame.shape[:2]
        scale = settings.frame_resize_width / w
        resized = cv2.resize(frame, (settings.frame_resize_width, int(h * scale)))

        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
            cv2.imwrite(tmp.name, frame)
            tmp_path = tmp.name

        try:
            count, density_map = self.density_estimator.estimate_with_heatmap(tmp_path)
        finally:
            os.remove(tmp_path)

        people_count = round(count)

        if people_count < 50:
            level = "LOW"
        elif people_count < 300:
            level = "MODERATE"
        else:
            level = "HIGH"

        # Normalize density map to 0-255 and resize to match the displayed image
        dm = density_map - density_map.min()
        if dm.max() > 0:
            dm = dm / dm.max()
        dm = (dm * 255).astype(np.uint8)
        dm_resized = cv2.resize(dm, (resized.shape[1], resized.shape[0]))

        heatmap = cv2.applyColorMap(dm_resized, cv2.COLORMAP_JET)
        overlay = cv2.addWeighted(resized, 0.55, heatmap, 0.45, 0)

        _, buffer = cv2.imencode(".jpg", overlay, [cv2.IMWRITE_JPEG_QUALITY, 85])
        annotated_b64 = base64.b64encode(buffer).decode("utf-8")

        return {
            "people_count": people_count,
            "density_level": level,
            "method": "density_regression",
            "annotated_image": annotated_b64,
        }