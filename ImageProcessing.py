import cv2
import tensorflow
import glob
import numpy

def load_files():
    fileList = []
    files = glob.iglob("trafficLightImages\*.jpg")
    for file in files:
        fileList.append(file)
    return fileList

#Loading the model (using pretrained modelZoo for basic img processing)
def load_model(path):
    model = tensorflow.saved_model.load(path)
    return model

#Combining image with tensor model processing
def stopSignDetection(model, img):
    tesnorInputFlow = img.copy() #Creates a copy of img data to process
    tesnorInputFlow = numpy.expand_dims(tesnorInputFlow, axis=0) #Expands the array for tensor flow
    detections = model(tesnorInputFlow) #Sends input to tensorflow model to detect if there is an image match
    return detections

#Puts boxes around stop signs if any
#TODO: Modify thresholds
def modifyImg(img, detections, threshold = 0.5):
    height, width, _ = img.shape #Gets dimensions of image
    for box, score, class_id in zip(detections['detection_boxes'][0], detections['detection_scores'][0],detections['detection_classes'][0]): #2 dimensional array looping
        if score >= threshold:
            ymin, xmin, ymax, xmax = box #Finding dimensions of box
            ymin, xmin, ymax, xmax = int(ymin * height), int(xmin * width), int(ymax * height), int(xmax * width) #Saling box dim to fit image

            # Class id 10 is Stoplight class id 12 is stopsign
            if class_id ==  12 or class_id == 10:
                cv2.rectangle(img, (xmin, ymin), (xmax, ymax), (255, 0, 0), 2)

#Currently created as a direcory
#TODO: Figure out to how stream these images from phone
files = load_files()
#MAKE SURE TO DOWNLOAD THIS AND ADD IT TO ROOT DIRECTORY (addded to git ignore due to restrictions)
#http://download.tensorflow.org/models/object_detection/tf2/20200711/ssd_resnet152_v1_fpn_640x640_coco17_tpu-8.tar.gz
#TODO Add download instructions to README if time permits 
modelPath = "Model\saved_model"
model = load_model(modelPath)
counter = 0

for file in files:
    stopSignImageNoPrep = cv2.imread(file)
    stopSignImagePrepped = stopSignImageNoPrep #TODO Prep the image to fti 612x612px

    detections = stopSignDetection(model, stopSignImagePrepped)
    modifyImg(stopSignImageNoPrep, detections)
    smaller = cv2.resize(stopSignImageNoPrep,(512,512))
    cv2.imshow("Stop sign detection", smaller)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    #Honestly I have no clue what this does, just saw it online lol
