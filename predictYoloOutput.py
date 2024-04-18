from ultralytics import YOLO
from ultralytics.solutions import object_counter
import cv2


model_path = r'C:\Users\yasha\OneDrive\Desktop\Desktop stuff\Interesting stuff\ASU 2nd sem\Perception\Project\Training\Output\train\weights\last.pt'
model = YOLO(model_path)

image_path = r'C:\Users\yasha\OneDrive\Desktop\Desktop stuff\Interesting stuff\ASU 2nd sem\Perception\Project\Images\person (10).png'

img = cv2.imread(image_path)
H, W, _ = img.shape


print("reading model\n-----------------\n")
results = model(img)
print("\n-----------------\nmodel read")

for result in results:
    for j, mask in enumerate(result.masks.data):
        mask = mask.numpy() * 255

        mask = cv2.resize(mask, (W, H))

        cv2.imwrite(
            r'C:\Users\yasha\OneDrive\Desktop\Desktop stuff\Interesting stuff\ASU 2nd sem\Perception\Project\Masks\person2.png', mask)
