from jetson_inference import detectNet
from jetson_utils import loadImage, saveImage
import jetson.inference
import jetson.utils

net = detectNet("ssd-mobilenet-v2", threshold=0.5)

img = loadImage("images/3.jpg")

detections = net.Detect(img)

print(f"Total objects detected: {len(detections)}\n")

for i, detection in enumerate(detections):
    classID = detection.ClassID
    confidence = detection.Confidence
    left = detection.Left
    top = detection.Top
    right = detection.Right
    bottom = detection.Bottom
    width = detection.Width
    height = detection.Height
    area = detection.Area
    center_x = detection.Center[0]
    center_y = detection.Center[1]
    class_name = net.GetClassDesc(classID)
    
    print(f"Detection {i+1}:")
    print(f"  ClassID: {classID}")
    print(f"  Class Name: {class_name}")
    print(f"  Confidence: {confidence:.2f}")
    print(f"  Left: {left:.2f}")
    print(f"  Top: {top:.2f}")
    print(f"  Right: {right:.2f}")
    print(f"  Bottom: {bottom:.2f}")
    print(f"  Width: {width:.2f}")
    print(f"  Height: {height:.2f}")
    print(f"  Area: {area:.2f}")
    print(f"  Center: ({center_x:.2f}, {center_y:.2f})")
    print("-" * 50)

saveImage("output.jpg", img)
print("\nDetection complete! Output saved to output.jpg")
