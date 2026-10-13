import numpy as np
import cv2
from PIL import Image
from lwcc import LWCC
from lwcc.util.functions import load_image

class DensityEstimator:
    def __init__(self):
        self.model = LWCC.load_model(model_name="CSRNet", model_weights="SHA")

    def estimate_with_heatmap(self, image_path):
        img, _ = load_image(image_path, "CSRNet", resize_img=False)
        img = img.to(next(self.model.parameters()).device)

        import torch
        with torch.no_grad():
            density_map = self.model(img)

        count = float(density_map.sum().item())
        density_map_np = density_map.squeeze().cpu().numpy()

        return count, density_map_np