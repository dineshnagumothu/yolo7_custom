import pandas as pd
import json
import cv2
import os
import numpy as np
from sklearn.model_selection import train_test_split
import shutil

def read_annotations(path):
    #annotations = pd.read_json(path)
    with open(path) as f:
        annotations = json.load(f)
    return annotations

def find_file(name, path):
    all_files = os.listdir(path)
    for file_name in all_files:
        if name in file_name:
            return os.path.join(path, file_name)

def get_all_images(annotations):
    all_ids = []
    all_image_file_names= []
    all_widths = []
    all_heights = []
    for img in annotations['images']:
        all_ids.append(img['id'])
        all_widths.append(img['width'])
        all_heights.append(img['height'])
        image_file_name = img['coco_url'].split('/')[-1]
        all_image_file_names.append(image_file_name)
    images = pd.DataFrame()
    images['id'] = all_ids
    images['file_name'] = all_image_file_names
    images['width'] = all_widths
    images['height'] = all_heights
    return images

def get_bounding_boxes_from_annotations(annotations):
    annotation_ids = []
    annotations_area = []
    annotatation_image_ids = []
    annotation_bbox = []
    for annotation in annotations['annotations']:
        area = annotation['bbox'][2]*annotation['bbox'][3]
        #if (area < 400):
        #    continue
        annotation_ids.append(annotation['id'])
        annotations_area.append(area)
        annotatation_image_ids.append(annotation['image_id'])
        annotation_bbox.append(annotation['bbox'])
    bboxes = pd.DataFrame()
    bboxes['bbox_id'] = annotation_ids
    bboxes['area'] = annotations_area
    bboxes['image_id'] = annotatation_image_ids
    bboxes['bbox'] = annotation_bbox
    return bboxes

def split_train_validation(df, test_split_ratio=0.1, random_state=42):
    img_train_ids, img_test_ids= train_test_split(df['image_id'].unique(), test_size=test_split_ratio, random_state=random_state)
    img_train_ids = img_train_ids.tolist()
    img_test_ids = img_test_ids.tolist()
    img_train = df[df['image_id'].isin(img_train_ids)]
    img_test = df[df['image_id'].isin(img_test_ids)]
    return img_train, img_test

def make_directory(parent_dir, directory):
    path = os.path.join(parent_dir, directory)
    try:
        os.makedirs(path, exist_ok = True)
        print("Directory '%s' created successfully" %path)
    except OSError as error:
        print("Directory '%s' can not be created")

def make_directories():
    make_directory("","custom_training")
    make_directory("custom_training","train")
    make_directory("custom_training","val")
    make_directory("custom_training/train","images")
    make_directory("custom_training/train","labels")
    make_directory("custom_training/val","images")
    make_directory("custom_training/val","labels")
    

def write_to_custom_training(df, path_type, copy_images=False):
    if copy_images!=True:
      print ("Not copying images")
    for i in range(len(df)):
        #Copy images
        if copy_images:
          shutil.copy('coco/images/val2017/'+df.iloc[i]['file_name'], "custom_training/"+path_type+"/images/")

        #write image filenames to file
        images_rel_path = 'images/'+path_type
        img_file_name = os.path.join(images_rel_path,df.iloc[i]['file_name'])

        #write labels as txt files
        label_file_name_only = df.iloc[i]['file_name'].split('.')[0]+'.txt'
        label_rel_path = 'custom_training/'+path_type+'/labels'
        label_file_name = os.path.join(label_rel_path, label_file_name_only)
        label = df.iloc[i]['label']
        x,y,w,h = df.iloc[i]['normalized_bbox']

        with open(label_file_name, "a") as label_file:
            label_file.write(str(label)+" "+"{:.3f}".format(x)+" "+"{:.3f}".format(y)+" "+"{:.3f}".format(w)+" "+"{:.3f}".format(h)+"\n")
    print ("Instances handled "+str(i+1))

def get_largest_smallest_boxes(df):
    #return df.groupby(['image_id']).agg({'area': [np.min,np.max]})
    max_area_indices = df.groupby('image_id')['area'].idxmax()
    min_area_indices = df.groupby('image_id')['area'].idxmin()
    common_indices = max_area_indices[max_area_indices.isin(min_area_indices)] #have to remove since no use
    indices_to_keep = max_area_indices.append(min_area_indices).tolist()
    for common_idx in common_indices:
        indices_to_keep = list(filter((common_idx).__ne__, indices_to_keep))
    result_df = df.loc[indices_to_keep]
    result_df['label'] = [1 if idx in max_area_indices.tolist() else 0 for idx in result_df.index]
    return result_df

def normalize_boxes(df, relative_coordinates=True):
    new_bboxes = []
    for i in range(len(df)):
        if relative_coordinates==False:
            bbox_x = df.iloc[i]['bbox'][0]/df.iloc[i]['width']
            bbox_y = df.iloc[i]['bbox'][1]/df.iloc[i]['height']
        else:
            bbox_x = (df.iloc[i]['bbox'][0] + df.iloc[i]['bbox'][2]/2) / df.iloc[i]['width']
            bbox_y = (df.iloc[i]['bbox'][1] + df.iloc[i]['bbox'][3]/2) / df.iloc[i]['height']
        bbox_width = df.iloc[i]['bbox'][2]/df.iloc[i]['width']
        bbox_height = df.iloc[i]['bbox'][3]/df.iloc[i]['height']
        new_bboxes.append([bbox_x,bbox_y, bbox_width, bbox_height])
    df['normalized_bbox'] = new_bboxes
    return df

if __name__ == '__main__':
    os.chdir("/content/")
    make_directories()
    annotations = read_annotations("coco/annotations/instances_val2017.json")
    images = get_all_images(annotations)
    bboxes = get_bounding_boxes_from_annotations(annotations)
    images_bboxes = pd.merge(images, bboxes,how='inner', left_on='id', right_on='image_id')
    selected_bboxes = get_largest_smallest_boxes(images_bboxes)
    normalized_bboxes = normalize_boxes(selected_bboxes)
    train,test = split_train_validation(normalized_bboxes, test_split_ratio=0.1, random_state=42)
    write_to_custom_training(train,'train', copy_images=True)
    write_to_custom_training(test,'val', copy_images=True)