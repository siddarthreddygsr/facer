# https://medium.com/@visrow/face-to-face-with-tomorrow-ai-enhanced-face-detection-for-the-modern-age-55bd37d0b1b4

import cv2 as cv2
from insightface.app import FaceAnalysis
import matplotlib.pyplot as plt
import time

app = FaceAnalysis(name='buffalo_l')

app.prepare(ctx_id=0)

image_path = "image.png"
image = cv2.imread(image_path)

start_time = time.time()
faces = app.get(image)
annotated_img = image.copy()
for face in faces:
    bbox = face.bbox.astype(int)
    cv2.rectangle(annotated_img, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 255, 0), 2)
end_time = time.time()
print(end_time - start_time)