import pandas as pd
import os
from data_folder_path import folder_path

orig_folder_path = folder_path+'Original_labels/'

dataframes = []

for filename in os.listdir(orig_folder_path):
    if filename.endswith('.csv'):
        file_path = os.path.join(orig_folder_path, filename)
        dataframes.append(pd.read_csv(file_path))

# Concatenate all dataframes into one
all_original = pd.concat(dataframes, ignore_index=True)

# all_original.to_csv(orig_folder_path + 'all_original.csv', index=False)

# Check for unique
is_unique = all_original['image_id'].sort_values().tolist() == sorted(list(set(all_original['image_id'])))
print(is_unique)