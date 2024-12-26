from data_folder_path import folder_path
import pandas as pd


image_label_map = pd.read_csv(folder_path + '/image_label_mapping.csv')

new_base_path = '/Users/joannelin/Data/LIDC'
image_label_map['image'] = image_label_map['image'].str.replace(r'^.*(?=/crops)', new_base_path, regex=True)

print(image_label_map['image'])

output_path = folder_path + '/image_label_mapping.csv'
image_label_map.to_csv(output_path, index=False)