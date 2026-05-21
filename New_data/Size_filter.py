import os

for i in range (8000):

    try:
        with open(f"annotations/{i}.txt") as file:
            data=file.readlines()
        coords=[int(x) for x in data[0].split(",")]
        if abs(coords[1]-coords[3])*abs(coords[0]-coords[2])<1000:
            os.remove(f"annotations/{i}.txt")
            os.remove(f"images_384_VarV2/{i}.jpg")
            print(f"файл {i}.txt удалён")
    except FileNotFoundError:
        continue


