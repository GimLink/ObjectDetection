# cvat xml to yolo

import os
import glob
import cv2
from xml.etree.ElementTree import parse

# for (path, dir, files) in os.walk('Python/0127/wine_dataset'):
#     for file in files:
#         ext = os.path.splitext(file)[-1]
#         print(ext)
# exit()

# xml 1 ~ 5
def find_xml_file(xml_folder_path) :
    all_root = []
    for (path, dir, files) in os.walk(xml_folder_path) :
        for filename in files :
            ext = os.path.splitext(filename)[-1]
            # ext -> .xml
            if ext == ".xml" :
                root = os.path.join(path, filename)
                # ./xml_data/test.xml
                all_root.append(root)
            else :
                pass
                

    return all_root

xml_folder_dir = "Python/0127/wine_dataset"
xml_paths = find_xml_file(xml_folder_dir)
# test = glob.glob(os.path.join(xml_paths[0], "*.xml"))


label_dict = {"AlcoholPercentage" : 0 , "Appellation AOC DOC AVARegion" : 1 , "Appellation QualityLevel":2, 
                "CountryCountry":3, "Distinct Logo":4, "Established YearYear":5,
              "Maker-Name":6, "Organic":7, "Sustainable":8, "Sweetness-Brut-SecSweetness-Brut-Sec":9, 
              "TypeWine Type":10, "VintageYear":11}

for xml_path in xml_paths :
    tree = parse(xml_path)
    root = tree.getroot()
    file_name = os.path.basename(xml_path).split('.xml')[0]
    object_metas = root.findall("object")
    size_meta = root.findall('size')
    img_width = int(size_meta[0].find("width").text)
    img_height = int(size_meta[0].find("height").text)
    for object_meta in object_metas :

        object_label = object_meta.find('name').text
        xmin = object_meta.find('bndbox').findtext('xmin')
        xmax = object_meta.find('bndbox').findtext('xmax')
        ymin = object_meta.find('bndbox').findtext('ymin')
        ymax = object_meta.find('bndbox').findtext('ymax')
        box = [int(xmin), int(ymin), int(xmax), int(ymax)]

        yolo_x = round(((box[0] + box[2])/2)/img_width, 6)
        yolo_y = round(((box[1] + box[3])/2)/img_height, 6)
        yolo_w = round((box[2] - box[0])/img_width, 6)
        yolo_h = round((box[3] - box[1])/img_height, 6)

        print("yolo xywh" , yolo_x, yolo_y, yolo_w, yolo_h)
        image_name_temp = file_name + '.txt'

        # txt file save folder
        os.makedirs("Python/0127/yolo_txt", exist_ok=True)

        # label
        label = label_dict[object_label]

        # txt save
        with open(f"Python/0127/yolo_txt/{image_name_temp}", 'a') as f :
            f.write(f"{label} {yolo_x} {yolo_y} {yolo_w} {yolo_h} \n")


