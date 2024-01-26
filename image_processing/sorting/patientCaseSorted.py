import os 
import shutil
import random
import pandas as pd

def patientSorted():
    image_path = r'D:\BreastCancer\RSNA_sorted'
    save_path = r'D:\BreastCancer\RSNA_PatientCaseSorted'
    for root, directories, files in os.walk(image_path):
        for file in files:
            patient_ID = file.split('_')[0]
            scr = os.path.join(root, file)
            
            folder_path = os.path.join(save_path, patient_ID)
            os.makedirs(folder_path , exist_ok=True)
            dst = os.path.join(folder_path, file)
            
            shutil.copy2(scr, dst)


def splitDataset():
# Paths
    src_path = "D:\\BreastCancer\\RSNA_PatientCaseSorted"
    dest_path = r'D:\BreastCancer\PatientCase_test'

    # List all folders in the source directory
    all_folders = [f for f in os.listdir(src_path) if os.path.isdir(os.path.join(src_path, f))]

    # Randomly select 20% of the folders
    num_to_select = int(0.20 * len(all_folders))
    folders_to_move = random.sample(all_folders, num_to_select)

    # Create destination directory if it doesn't exist
    if not os.path.exists(dest_path):
        os.makedirs(dest_path)

    # Move the selected folders
    for folder in folders_to_move:
        shutil.move(os.path.join(src_path, folder), os.path.join(dest_path, folder))

    print(f"Moved {num_to_select} folders from {src_path} to {dest_path}")


def CalculatePositive(folder_name):
    positiveCase = 0
    totalFile = 0
    for root, directories, files in os.walk(folder_name):
        totalFile+=len(files)
        for file in files:
            number = file.split('_')[1].split('.')[0]
            if isCancer(number): 
                positiveCase+=1 
            # print(positiveCase)


    return totalFile,positiveCase




def isCancer(image_id):
    train_csv = r'C:\Users\Woody\Desktop\metaData\train.csv'
    df = pd.read_csv(train_csv)
    filtered_df = df[df['image_id'] == int(image_id)]

    if not filtered_df.empty:
        cancer = filtered_df['cancer'].iloc[0]
        return True if cancer == 1 else False

def main():
    train = r'D:\BreastCancer\PatientCase_train'
    test = r'D:\BreastCancer\PatientCase_test'
    print(CalculatePositive(train))
    print(CalculatePositive(test))

main()