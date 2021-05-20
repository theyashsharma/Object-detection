from imageai.Detection import ObjectDetection
import os
execution_path = os.getcwd()
detector=ObjectDetection()
detector.setModelTypeAsRetinaNet()
detector.setModelPath( os.path.join(execution_path , "resnet50_coco_best_v2.0.1.h5"))
detector.loadModel()
custom_objects = detector.CustomObjects(person=True, car=False, banana=False)
detections = detector.detectCustomObjectsFromImage(input_image=os.path.join(execution_path , "kanwar.jpg"), output_image_path=os.path.join(execution_path , "kanwar_new.jpg"), custom_objects=custom_objects, minimum_percentage_probability=15)
count=0
for eachObject in detections:
   print(eachObject["name"] , " : " , eachObject["percentage_probability"] )
   print("--------------------------------")
   count=count+1
print(count)
