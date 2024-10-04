import os
import sys
import cv2
from ultralytics import YOLO
import numpy as np

current_dir=sys.path[0]
os.chdir(current_dir)
# print(os.getcwd())
#files = os.listdir(os.curdir)  
#print(files)

img_name = 'infflag1'
img_path = current_dir+'/'+img_name+'.png'
model = YOLO(current_dir+'/flag_best_weight.pt')
#model = YOLO("yolov8n-cls.pt")  # load a pretrained model (recommended for training)
# Force the model to run on CPU
model.to('cpu')

img_cv = cv2.imread(img_path)


# Run the model
results = model(img_path)

bounding_box = []

for result in results:
    bounding_box.append(result.boxes.xyxy)
print(results[0].boxes.conf)
        

bounding_box = np.asarray(bounding_box[0][0])

x_min, y_min, x_max, y_max = bounding_box.astype(int)
cv2.rectangle(img_cv, (x_min, y_min), (x_max, y_max), (255, 0, 0), 2)  # Draw rectangle with blue color and thickness 2


# # Display the result
cv2.imshow('image',img_cv)

cv2.waitKey(0)
cv2.destroyAllWindows()

