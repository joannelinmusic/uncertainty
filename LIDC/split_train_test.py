from data_folder_path import folder_path
import pandas as pd
from sklearn.model_selection import train_test_split

image_label_map = pd.read_csv(folder_path + '/image_label_mapping.csv')

agreement_groups = image_label_map.groupby('agreement')

for agreement, group in agreement_groups:
    # Shuffle and split the data
    train, test = train_test_split(group, test_size=0.3, random_state=42, shuffle=True)
    
    # Determine file names based on agreement
    if agreement == 3:
        train_file = "full_train_LIDC.csv"
        test_file = "full_test_LIDC.csv"
    elif agreement == 2:
        train_file = "high_train_LIDC.csv"
        test_file = "high_test_LIDC.csv"
    elif agreement == 1:
        train_file = "low_train_LIDC.csv"
        test_file = "low_test_LIDC.csv"
    elif agreement == 0:
        train_file = "no_train_LIDC.csv"
        test_file = "no_test_LIDC.csv"

    train.to_csv(folder_path + '/model_ready_dataset/' + train_file, index=False)
    test.to_csv(folder_path + '/model_ready_dataset/' + test_file, index=False)