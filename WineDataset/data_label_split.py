import os
import glob
import random
import shutil


def test_val_split(file_path):
    all_path = glob.glob(os.path.join(file_path, '*.jpg'))
    os.makedirs('Python/0127/dataset/train/images', exist_ok=True)
    for i in random.sample(all_path, int(0.8 * len(all_path))) :
        shutil.move(i, 'Python/0127/dataset/train/images/')
    rest_path = glob.glob(os.path.join(file_path, '*.jpg'))
    os.makedirs('Python/0127/dataset/val/images', exist_ok=True)
    for i in random.sample(rest_path, int(0.5 * len(rest_path))) :
        shutil.move(i, 'Python/0127/dataset/val/images/')
    test_path = glob.glob(os.path.join(file_path, '*.jpg'))
    os.makedirs('Python/0127/dataset/test/images', exist_ok=True)
    for i in test_path :
        shutil.move(i, 'Python/0127/dataset/test/images/')

def label_split(label_path, train_path, val_path, test_path):
    train_list, val_list, test_list = [], [], []

    train_labels = os.listdir(train_path)
    os.makedirs('Python/0127/dataset/train/labels', exist_ok=True)
    for train_label in train_labels :
        train_label_name = train_label.split('.jpg')[0]
        train_list.append(train_label_name)

    val_labels = os.listdir(val_path)
    os.makedirs('Python/0127/dataset/val/labels', exist_ok=True)
    for val_label in val_labels :
        val_label_name = val_label.split('.jpg')[0]
        val_list.append(val_label_name)

    test_labels = os.listdir(test_path)
    os.makedirs('Python/0127/dataset/test/labels', exist_ok=True)
    for test_label in test_labels :
        test_label_name = test_label.split('.jpg')[0]
        test_list.append(test_label_name)

    all_labels = os.listdir(label_path) 
    label_names = []
    for i in all_labels :
        label_names.append(i.split('.txt')[0])
    for label in label_names :
        if label in train_list:
            shutil.move(label_path + f'/{label}.txt', 'Python/0127/dataset/train/labels/')
        elif label in val_list:
            shutil.move(label_path + f'/{label}.txt', 'Python/0127/dataset/val/labels/')
        else:
            shutil.move(label_path + f'/{label}.txt', 'Python/0127/dataset/test/labels/')
    

    


# test_val_split('Python/0127/wine_dataset')
label_split('Python/0127/yolo_txt', 'Python/0127/dataset/train/images', 
            'Python/0127/dataset/val/images', 'Python/0127/dataset/test/images')