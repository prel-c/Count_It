import json
import os

with open('annotation_FSC147_384.json') as f:
    annotations=json.load(f)

with open('Train_Test_Val_FSC_147.json') as f:
    Task=json.load(f)

test=Task["test"]

os.makedirs("annotations", exist_ok=True)

for i in range (8000):
    filename=f"{i}.jpg"
    if filename in annotations and filename in test:
        data=annotations[filename]
        box=data['box_examples_coordinates'][1]
        left_bottom=str(box[0][0])+","+str(box[0][1])
        right_top=str(box[2][0])+","+str(box[2][1])
        count=len(data['points'])
        with open(f'annotations/{i}.txt', 'w', encoding='utf-8') as file:
            file.write(left_bottom+","+right_top+"\n")
            file.write(str(count))
        print(f"Создан: {i}.txt' ")
    else:
        print("f")
