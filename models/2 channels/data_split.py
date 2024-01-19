import os
import shutil
import random
import pandas as pd

def move_random_folders(source_directory, destination_directory, percentage=0.2):
    if not os.path.exists(destination_directory):
        os.makedirs(destination_directory)

    all_folders = [f for f in os.listdir(source_directory) if os.path.isdir(os.path.join(source_directory, f))]
    number_to_move = int(len(all_folders) * percentage)
    folders_to_move = random.sample(all_folders, number_to_move)

    for folder in folders_to_move:
        src_folder = os.path.join(source_directory, folder)
        dest_folder = os.path.join(destination_directory, folder)
        shutil.move(src_folder, dest_folder)
        print(f"Moved {folder} from {source_directory} to {destination_directory}")

source_directory = r'/gpfs/home/hluo/Honros/BreastDensity/Vindir'
destination_directory = r'/gpfs/home/hluo/Honros/BreastDensity/Vindir_test'

move_random_folders(source_directory, destination_directory)
