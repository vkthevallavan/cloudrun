import ast

import torchvision.transforms as transforms
from torch.autograd import Variable
import torchvision.models as models
from torch import __version__
from PIL import Image
import urllib.request
import io

alexnet = models.alexnet(pretrained=True)

model_name = 'alexnet'
models = {'alexnet': alexnet}

# obtain ImageNet labels
with open('imagenet1000_clsid_to_human.txt') as imagenet_classes_file:
    imagenet_classes_dict = ast.literal_eval(imagenet_classes_file.read())

def classifier(img_path):
    f= ''
    with urllib.request.urlopen(img_path) as url:
        f = io.BytesIO(url.read())
        print('f1',f)

    print('f2',f)
    img_pil = Image.open(f)
    # load the image
    #img_pil = Image.open(img_path)
    print("image path",img_path)
    print("img_pil",img_pil)
    # define transforms
    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # preprocess the image
    img_tensor = preprocess(img_pil)
    #print("img_tensor",img_tensor)
    # resize the tensor (add dimension for batch)
    img_tensor.unsqueeze_(0)
    #print("img_tensor2",img_tensor)
    # wrap input in variable, wrap input in variable - no longer needed for
    # v 0.4 & higher code changed 04/26/2018 by Jennifer S. to handle PyTorch upgrade
    pytorch_ver = __version__.split('.')
    
    # pytorch versions 0.4 & hihger - Variable depreciated so that it returns
    # a tensor. So to address tensor as output (not wrapper) and to mimic the 
    # affect of setting volatile = True (because we are using pretrained models
    # for inference) we can set requires_gradient to False. Here we just set 
    # requires_grad_ to False on our tensor 
    if int(pytorch_ver[0]) > 0 or int(pytorch_ver[1]) >= 4:
        img_tensor.requires_grad_(False)
        #print("img_tensor3",img_tensor)
    # pytorch versions less than 0.4 - uses Variable because not-depreciated
    else:
        # apply model to input
        # wrap input in variable
        data = Variable(img_tensor, volatile = True) 
        #print("img_tensor4",img_tensor)
        #print("data",data)

    # apply model to input
    model = models[model_name]
    
    # puts model in evaluation mode
    # instead of (default)training mode
    model = model.eval()
    print('pytorch_ver eval ',pytorch_ver)
    # apply data to model - adjusted based upon version to account for 
    # operating on a Tensor for version 0.4 & higher.
    if int(pytorch_ver[0]) > 0 or int(pytorch_ver[1]) >= 4:
        print('came here1')
        output = model(img_tensor)

    # pytorch versions less than 0.4
    else:
        print('came here2')
        # apply data to model
        output = model(data)
    #print('output',output)  
    #print('output data',output.data)  
    #print('output data numpy',output.data.numpy())  
    # return index corresponding to predicted class
    pred_idx = output.data.numpy().argmax()
    print('pred_idx is',pred_idx)  
    return imagenet_classes_dict[pred_idx]