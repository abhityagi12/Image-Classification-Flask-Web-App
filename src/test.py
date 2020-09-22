import torch
from torchvision import datasets,transforms
from PIL import Image
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


train_path='/home/abhinav/kaggle/intel_image_classification/input/train/'
val_path='/home/abhinav/kaggle/intel_image_classification/input/val/'

saving_path={
    'mobile_net':"/home/abhinav/kaggle/intel_image_classification/models/mobilenetv2.pth",
    'resnet50':"/home/abhinav/kaggle/intel_image_classification/models/resnet50.pth",
    'SpinalNet_ResNet':"/home/abhinav/kaggle/intel_image_classification/models/SpinalNet_ResNet.pth"
}



dataset = {
    'train': datasets.ImageFolder(root=train_path, transform=image_transforms['train']),
    'valid': datasets.ImageFolder(root=val_path, transform=image_transforms['valid'])
}
 
dataset_sizes = {
    'train':len(dataset['train']),
    'valid':len(dataset['valid'])
}

dataloaders = {
    'train':torch.utils.data.DataLoader(dataset['train'], batch_size=32, shuffle=True),
    'valid':torch.utils.data.DataLoader(dataset['valid'], batch_size=32, shuffle=True,)
}

class_names = dataset['train'].classes
print("Classes:", class_names)




test_path='/home/abhinav/kaggle/intel_image_classification/input/test/'

sample_image=test_path+'572.jpg'

model_path='/home/abhinav/kaggle/intel_image_classification/models/resnet50.pth'

device=torch.device("cuda:0" if torch.cuda.is_available() else 'cpu')


model_ft=torch.load(model_path)
model_ft.to(device)

model_ft.eval()

image=Image.open(sample_image)

def predict_image(image):
    image_tensor = image_transforms['valid'](image).float()
    image_tensor = image_tensor.unsqueeze_(0)
    input = torch.autograd.Variable(image_tensor)
    input = input.to(device)
    output = model_ft(input)
    index = output.data.cpu().numpy().argmax()
    return index

print(predict_image(image))

# model_path="/home/abhinav/kaggle/intel_image_classification/models/mobilenetv2.pth"

# dataset = {
#     'test': datasets.ImageFolder(root=test_path, transform=image_transforms['valid'])
# }
 
# dataset_sizes = {
#     'test':len(dataset['test']),
# }

# dataloaders = {
#     'test':torch.utils.data.DataLoader(dataset['test'], batch_size=32, shuffle=True),
# }

# class_names = dataset['test'].classes
# print("Classes:", class_names)
 
# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# print(device," is available")

# model_ft=torch.load(model_path)
# model_ft = model_ft.to(device)

# print('Model loaded to device......\n')

# def check_accuracy(loader, model):
#     num_correct = 0
#     num_samples = 0
#     model.eval()
    
#     with torch.no_grad():
#         for x, y in loader["test"]:
#             x = x.to(device=device)
#             y = y.to(device=device)
            
#             scores = model(x)
#             _, predictions = scores.max(1)
#             num_correct += (predictions == y).sum()
#             num_samples += predictions.size(0)
        
#         print(f'Got {num_correct} / {num_samples} with accuracy {float(num_correct)/float(num_samples)*100:.2f}') 
    
#     model.train()

# check_accuracy(dataloaders,model_ft)