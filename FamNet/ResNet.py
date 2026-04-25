import torch
import torchvision.models as models
import torchvision.transforms as transforms
import torch.nn.functional as F
import cv2 as cv

image = cv.imread("image.png")
image = cv.cvtColor(image, cv.COLOR_BGR2RGB)

weights = models.ResNet50_Weights.DEFAULT
categories = weights.meta["categories"]

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize((224, 224)),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

image = transform(image).unsqueeze(0)

resnet50 = models.resnet50(weights = weights)
resnet50.eval()
result = resnet50(image)

probs = F.softmax(result, dim=1)

top_probs, top_idxs = torch.topk(probs, 5)

for p, idx in zip(top_probs[0], top_idxs[0]):
    print(categories[idx], p.item())
