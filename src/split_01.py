from logging import root
import shutil
import os 
import argparse
import yaml
import pandas as pd
import numpy as np 
import random
from get_data import get_data , read_params

def train_and_test(config_path):
    config=get_data(config_path)
    print(f"Config Data: {config}")
    print(f"Raw Data Section: {config.get('raw_data', 'Missing')}")
    print(f"Raw Data Source: {config.get('raw_data', {}).get('data_src', 'Missing')}")

    root_dir=config["raw_data"]["data_src"]
    dest=config["load_data"]["preprocessed_data"]
    os.makedirs(os.path.join(dest,"train"),exist_ok=True)
    os.makedirs(os.path.join(dest,"test"),exist_ok=True)
    classes = ['no_tumor', 'pituitary_tumor', 'meningioma_tumor', 'glioma_tumor']
    for class_name in classes:
        os.makedirs(os.path.join(dest, 'train', class_name), exist_ok=True)
        os.makedirs(os.path.join(dest, 'test', class_name), exist_ok=True)
    training_dir=os.path.join(root_dir ,"Training")
    for class_name in classes:
        src_dir=os.path.join(training_dir,class_name)
        if not os.path.exists(src_dir):
            print(f"Warning: Directory {src_dir} does not exist. Skipping...")
            continue
        files=os.listdir(src_dir)
        print(f"{class_name} (Training) -> {len(files)} images")
        for f in files:
            src_path=os.path.join(src_dir,f)
            dst_path=os.path.join(dest,"train",class_name,f)
            shutil.copy(src_path,dst_path)
        print(f"Done copying training data for {class_name}")

    testing_dir = os.path.join(root_dir, 'Testing')
    for class_name in classes:
            src_dir = os.path.join(testing_dir, class_name)
            if not os.path.exists(src_dir):
                print(f"Warning: Directory {src_dir} does not exist. Skipping...")
                continue
            
            files = os.listdir(src_dir)
            print(f"{class_name} (Testing) -> {len(files)} images")
        
            for f in files:
                src_path = os.path.join(src_dir, f)
                dst_path = os.path.join(dest, 'test', class_name, f)
                shutil.copy(src_path, dst_path)
            
            print(f"Done copying testing data for {class_name}")
 
 
if __name__=='__main__':
    args=argparse.ArgumentParser()
    args.add_argument('--config',default='params.yaml')
    passed_args=args.parse_args()
    train_and_test(config_path=passed_args.config)  
  


