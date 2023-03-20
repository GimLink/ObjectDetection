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
# def find_xml_file(xml_folder_path) :
#     all_root = []
#     for (path, dir, files) in os.walk(xml_folder_path) :
#         for filename in files :
#             ext = os.path.splitext(filename)[-1]
#             # ext -> .xml
#             if ext == ".xml" :
#                 root = os.path.join(path, filename)
#                 # ./xml_data/test.xml
#                 all_root.append(root)
#             else :
#                 pass
                

#     return all_root

xml_folder_dir = "Python/0131/animals/test"
xml_paths = glob.glob(os.path.join(xml_folder_dir, '*.xml'))
# print(xml_paths)



label_dict = {"cat" : 0 , "chicken" : 1, "cow" : 2, "dog" : 3, "fox" : 4, "goat" : 5, "horse" : 6, "person" : 7, "racoon" : 8, "skunk" : 9}

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
        os.makedirs("Python/0131/test_labels", exist_ok=True)

        # label
        label = label_dict[object_label]

        # txt save
        with open(f"Python/0131/test_labels/{image_name_temp}", 'a') as f :
            f.write(f"{label} {yolo_x} {yolo_y} {yolo_w} {yolo_h} \n")


