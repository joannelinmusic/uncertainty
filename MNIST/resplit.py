import pandas as pd
import os
import numpy as np
from data_folder_path import folder_path

all_data = pd.read_csv(folder_path + 'all_original.csv')

all_data_shuffled = all_data.iloc[np.random.permutation(len(all_data))].reset_index(drop=True)
group_size = len(all_data_shuffled) // 5

groups = []
for i in range(5):
    group_data = all_data_shuffled.iloc[i * group_size:(i + 1) * group_size]
    group_data['agreement'] = i
    groups.append(group_data)

all_data_with_agreement = pd.concat(groups)

# Save the modified DataFrame back to a CSV file
# all_data_with_agreement.to_csv(folder_path + 'all_original_with_agreement.csv', index=False)

agreement_counts = all_data_with_agreement['agreement'].value_counts()
# print(agreement_counts)

def split_five_agreement():
    for agreement_value in all_data_with_agreement['agreement'].unique():
        subset = all_data_with_agreement[all_data_with_agreement['agreement'] == agreement_value]
        file_path = folder_path + 'five_raters/' + f'agreement_{agreement_value}.csv'
        
        # Save the subset to a CSV file
        subset.to_csv(file_path, index=False)
        print(f"Saved subset with agreement {agreement_value} to {file_path}")


# Train test split
five_rater_folder = folder_path + 'five_raters/'
csv_files = [f for f in os.listdir(five_rater_folder) if f.endswith('.csv')]
for filename in csv_files:
    data_path = five_rater_folder + filename
    print(data_path)
    data = pd.read_csv(data_path)

    data_shuffled = data.sample(frac=1).reset_index(drop=True)
    split_index = int(0.7 * len(data_shuffled))  # 70% for training
    train_set = data_shuffled[:split_index]
    test_set = data_shuffled[split_index:]

    base_filename = os.path.splitext(filename)[0]
    train_file_path = os.path.join(five_rater_folder, f'{base_filename}_train.csv')
    train_set.to_csv(train_file_path, index=False)
    
    test_file_path = os.path.join(five_rater_folder, f'{base_filename}_test.csv')
    test_set.to_csv(test_file_path, index=False)