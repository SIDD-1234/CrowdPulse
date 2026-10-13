from lwcc import LWCC

img_path = "/path/to/that/dense_crowd_image.jpg"

count = LWCC.get_count(img_path, model_name="CSRNet", model_weights="SHA", resize_img=False)
print(f"CSRNet (Shanghai A weights) estimated count: {count}")
