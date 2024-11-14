import os
import shutil
import random

def split_dataset(dataset_dir, output_dir, train_ratio=0.8):
    # Create output directories
    train_dir = os.path.join(output_dir, 'train')
    test_dir = os.path.join(output_dir, 'test')
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)

    # Iterate over each class folder
    for class_name in os.listdir(dataset_dir):
        class_dir = os.path.join(dataset_dir, class_name)
        if os.path.isdir(class_dir):
            # Create class subdirectories in train and test folders
            train_class_dir = os.path.join(train_dir, class_name)
            test_class_dir = os.path.join(test_dir, class_name)
            os.makedirs(train_class_dir, exist_ok=True)
            os.makedirs(test_class_dir, exist_ok=True)

            # List all files in the class directory
            files = os.listdir(class_dir)
            random.shuffle(files)

            # Split files into train and test sets
            split_index = int(len(files) * train_ratio)
            train_files = files[:split_index]
            test_files = files[split_index:]

            # Copy files to the respective directories
            for file_name in train_files:
                shutil.copy(os.path.join(class_dir, file_name), train_class_dir)
            for file_name in test_files:
                shutil.copy(os.path.join(class_dir, file_name), test_class_dir)

    print(f'Dataset split into train and test sets with {train_ratio*100}% for training.')

# Example usage
dataset_dir = 'dataset/EuroSAT_RGB/'
output_dir = 'dataset/EuroSAT_RGB_split/'
split_dataset(dataset_dir, output_dir)