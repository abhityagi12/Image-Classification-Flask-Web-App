from flask import Flask
from flask import render_template
from flask import request
import os
import torch
from torchvision import datasets,transforms
from PIL import Image


app=Flask(__name__)
UPLOAD_FOLDER="/home/abhinav/kaggle/intel_image_classification/static/media/"


model_path='/home/abhinav/kaggle/intel_image_classification/models/resnet50.pth'

device=torch.device("cuda:0" if torch.cuda.is_available() else 'cpu')

image_transforms = { 
    'train': transforms.Compose([
        transforms.RandomResizedCrop(size=256, scale=(0.8, 1.0)),
        transforms.RandomRotation(degrees=15),
        transforms.CenterCrop(size=224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ]),
    'valid': transforms.Compose([
        transforms.Resize(size=256),
        transforms.CenterCrop(size=224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])
}


def predict(image,model):
	image=Image.open(image)
	image_tensor = image_transforms['valid'](image).float()
	image_tensor = image_tensor.unsqueeze_(0)
	input = torch.autograd.Variable(image_tensor)
	input = input.to(device)
	output = model_ft(input)
	index = output.data.cpu().numpy().argmax()
	return index


@app.route("/", methods=["GET","POST"])
def upload_predict():
	if request.method == "POST":
		image_file=request.files["image"]
		if image_file:
			image_location=os.path.join(
				UPLOAD_FOLDER,
				image_file.filename
			)
			image_file.save(image_location)
			pred=predict(image_location,model_ft)
			print(pred)
			if pred==0:
				pred="buildings"
			elif pred==1:
				pred="forest"
			elif pred==2:
				pred="glacier"
			elif pred==3:
				pred="mountain"
			elif pred==4:
				pred="sea"
			elif pred==5:
				pred="street"
			return render_template("index.html",prediction=pred,image_loc=image_file.filename)
	return render_template("index.html",prediction=0, image_loc=None)

if __name__ == '__main__':
	model_ft=torch.load(model_path)
	model_ft.to(device)
	model_ft.eval()
	app.run(port=12000, debug=True)