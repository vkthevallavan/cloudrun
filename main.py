#from classifier import classifier 
import json
import pickle
import numpy
import torch
from torchvision import models
from torch import nn
import torchvision.transforms as transforms
import torchvision.models as models
import ast
from datetime import datetime

orignal_models = models.alexnet(pretrained=True)    
class AlexnetConvLast(nn.Module):
    def __init__(self ):
        super(AlexnetConvLast, self).__init__()
        #self.__class__(output)
        
        self.features = nn.Sequential(
            *list(orignal_models.features.children())[1:]
        )
        self.avgpool = nn.AdaptiveAvgPool2d(output_size=(6,6))
        self.classifier = nn.Sequential(*list(orignal_models.classifier.children()))
    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return (x)


def hello_world(request):
    startTime = datetime.now()
    print("Start Time request came1:", startTime)
    """Responds to any HTTP request.
    Args:
        request (flask.Request): HTTP request object.
    Returns:
        The response text or any set of values that can be turned into a
        Response object using
        `make_response <http://flask.pocoo.org/docs/1.0/api/#flask.Flask.make_response>`.
    """
    #model_name = 'alexnet'

    #img_path= "https://res.cloudinary.com/demo/video/upload/dog.jpg"
    request_json = request.get_json(force=True)
    
    message = request_json['message']
    
    #decodedArrays = json.loads(request_json)
    #print('came here 2',message)
    finalNumpyArray = numpy.asarray(request_json['message'])
    #print("NumPy Array")
    #print(finalNumpyArray)    
    torchobject = torch.Tensor(finalNumpyArray)
    modelStartTime = datetime.now()
    print("Model start Time request came1:", modelStartTime)
    modelLast = AlexnetConvLast()
    modelLast = modelLast.eval()    
    outputMerge = modelLast(torchobject)
    modelEndTime = datetime.now()
    print("Model Time request end time:", modelEndTime)
    difference = modelStartTime - modelEndTime;
    print("Model difference",difference.total_seconds())
    pred_idx = outputMerge.data.numpy().argmax()
    
    #print('pred_idx is',pred_idx)
    with open('imagenet1000_clsid_to_human.txt') as imagenet_classes_file:
        imagenet_classes_dict = ast.literal_eval(imagenet_classes_file.read())  
    
    image_classification = imagenet_classes_dict[pred_idx]
    #print('output',image_classification)
    result = "Image was was classified as: " + image_classification;
    endTime = datetime.now()
    print("Time request end time:", endTime)
    difference = startTime - endTime;
    print("difference",difference.total_seconds())
    #image_classification = classifier(messageUrl)

    # prints result from running classifier() function
    #print("\nResults from test_classifier.py\nImage:", messageUrl, "using model:",model_name, "was classified as a:", image_classification)
    #result = "Classified image in path" + img_path + "using model:" + model_name + " was classified as: " + image_classification;
    return str(pred_idx)