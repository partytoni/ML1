from PIL import Image
import os

augmented_dir = "aug-set/"

classes = ["HAZE", "SUNNY", "RAINY", "SNOWY"]

for elem in classes:
    print("class:", elem)
    extended_path = augmented_dir+elem

    for image_path in os.listdir(extended_path):
        full_path = extended_path+"/"+image_path
        try:
            img = Image.open(full_path)
        except Exception as e:
            print(e)
