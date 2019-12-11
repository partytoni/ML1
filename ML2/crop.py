from PIL import Image
import os

train_data_dir = "MWI-Dataset/"  # 400 images folder
augmented_dir = "aug-set/"

classes = ["HAZE", "SUNNY", "RAINY", "SNOWY"]

for elem in classes:
    print("class:", elem)
    extended_path = train_data_dir+elem

    for image_path in os.listdir(extended_path):
        full_path = extended_path+"/"+image_path
        with Image.open(full_path) as img:
            width, height = img.size
            if width<=height:
                minor = width
            else:
                minor = height
            
            new_width = minor
            new_height = minor

            left = (width - new_width)/2
            top = (height - new_height)/2
            right = (width + new_width)/2
            bottom = (height + new_height)/2

            img = img.crop((left, top, right, bottom))
            
            try:
                img.convert("RGB")
                img.save(augmented_dir+elem+"/"+image_path)
            except OSError as e:
                print(image_path, e)    
