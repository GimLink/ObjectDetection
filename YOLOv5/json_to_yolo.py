import json
import os
import glob
import cv2



def label_image(json_path, image_path):
    with open(json_path, 'r') as j :
        wine_data = json.load(j)


    # make label dict
    all_category = wine_data['categories']
    label_list = []
    for label in all_category:
        label_list.append(label['name'])

    label_dict = {}
    for i, label in enumerate(label_list):
        label_dict[i] = label

    all_image_path = glob.glob(os.path.join(image_path, '*.jpg'))
    
    annotations  = wine_data['annotations']
    # len(wine_data['images']) == len(all_image_path)
    image_dict = {}
    for i in range(len(all_image_path)):
        image_id = wine_data['images'][i]['id']
        image_name = wine_data['images'][i]['file_name']
        image_height = wine_data['images'][i]['height']
        image_width = wine_data['images'][i]['width']
        image_dict[image_id] = [image_name, image_height, image_width]
       

    

    for i in range(len(annotations)):
        id = wine_data['annotations'][i]['image_id']
        image_name = image_dict[id][0]
        image_height = image_dict[id][1]
        image_width = image_dict[id][2]
        txt_name = image_name.replace('.jpg', '.txt')
        label_id = wine_data['annotations'][i]['category_id']
        bbox = wine_data['annotations'][i]['bbox']
        x, y, w, h = bbox
        x1 = int(x)
        y1 = int(y)
        x2 = int(w + x)
        y2 = int(h + y)

        yolo_x = round(((x1 + x2) / 2) / int(image_width), 6)
        yolo_y = round(((y1 + y2) / 2) / int(image_height), 6)
        yolo_w = round((x2 - x1) / int(image_width), 6)
        yolo_h = round((y2 - y1) / int(image_height), 6)

        imgPath = f'python/0130//wine_labels_coco/train/{image_name}'
        # print(imgPath)
        img = cv2.imread(imgPath)
        # print(img)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        cv2.rectangle(img,
                      (int(x), int(y)),
                      (int(x + w), int(y + h)),
                      (0, 0, 255), 2)
        
        cv2.putText(img, label_dict[label_id],
                    (int(x), int(y - 5)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        cv2.imshow('test', img)
        cv2.waitKey(0)
        
        
        os.makedirs('python/0130/json_label', exist_ok=True)

        # txt save
        with open(f"python/0130/json_label/{txt_name}", 'a') as f:
            f.write(f"{label_id} {yolo_x} {yolo_y} {yolo_w} {yolo_h}\n")

    


        
        
        
    
    
        


json_path = 'Python/0130/wine_labels_coco/train/_annotations.coco.json'
image_path = 'Python/0130/wine_labels_coco/train'

label_image(json_path, image_path)

