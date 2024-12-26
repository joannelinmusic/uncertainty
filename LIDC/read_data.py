from data_folder_path import folder_path
import pandas as pd

# amal
full_test_data = pd.read_csv(folder_path + 'model_ready_dataset/' + 'full_test_LIDC.csv')
full_train_data = pd.read_csv(folder_path + 'model_ready_dataset/' + 'full_train_LIDC.csv')
high_test_data = pd.read_csv(folder_path + 'model_ready_dataset/' + 'high_test_LIDC.csv')
high_train_data = pd.read_csv(folder_path + 'model_ready_dataset/' + 'high_train_LIDC.csv')
low_test_data = pd.read_csv(folder_path + 'model_ready_dataset/' + 'low_test_LIDC.csv')
low_train_data = pd.read_csv(folder_path + 'model_ready_dataset/' + 'low_train_LIDC.csv')
no_test_data = pd.read_csv(folder_path + 'model_ready_dataset/' + 'no_test_LIDC.csv')
no_train_data = pd.read_csv(folder_path + 'model_ready_dataset/' + 'no_train_LIDC.csv')

