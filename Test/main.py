import cv2
from test import test

k=0
sae=0
sse=0
sape=0


for i in range(8000):
    try:
        with open(f"/mnt/c/CV/FSC147_modifications/FSC147/annotations/{i}.txt", 'r') as file:
            data=file.readlines()
        coords=[int(x) for x in data[0].split(",")]
        count=int(data[1])
        image_BGR=cv2.imread(f"/mnt/c/CV/FSC147_modifications/FSC147/images_384_VarV2/{i}.jpg")
        image1=image_BGR.copy()
        image=cv2.cvtColor(image_BGR, cv2.COLOR_BGR2GRAY)
        template=image[coords[1]:coords[3], coords[0]:coords[2]]
        result, image2=test(image, template, image1)
        cv2.imwrite(f"results/{i}.jpg", image2)

        k=k+1
        error=abs(result-count)
        sae+=error
        sse+=error**2
        sape+=error/count
        print(k)
    except FileNotFoundError:
        continue

MAE=sae/k
RMSE=(sse/k)**0.5
MAPE=sape/k*100

print(f"\nOn test data, MAE: {MAE:.2f}, RMSE: {RMSE:.2f}, MAPE: {MAPE:.2f}%")
