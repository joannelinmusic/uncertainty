import pandas as pd
import numpy as np
import csv
import os
from data_folder_path import folder_path
from read_data import agreement_0_train_data, agreement_1_train_data, agreement_2_train_data, agreement_3_train_data, agreement_4_train_data, agreement_0_test_data, agreement_1_test_data, agreement_2_test_data, agreement_3_test_data, agreement_4_test_data

# Define probabilities for each number from 0 to 9
probabilities_lists = {
0: [0, 0.03, 0.08, 0.08, 0.05, 0.08, 0.16, 0.06, 0.38, 0.08],
1: [0.09, 0, 0.03, 0.12, 0.17, 0.03, 0.03, 0.34, 0.07, 0.12],
2: [0.14, 0.02, 0, 0.2, 0.04, 0.08, 0.14, 0.05, 0.25, 0.08],
3: [0.09, 0.05, 0.14, 0, 0.07, 0.14, 0.09, 0.1, 0.18, 0.14],
4: [0.07, 0.09, 0.04, 0.09, 0, 0.09, 0.07, 0.06, 0.12, 0.37],
5: [0.09, 0.01, 0.05, 0.13, 0.07, 0, 0.33, 0.03, 0.16, 0.13],
6: [0.132908028,0.009037746,0.070707071,0.070707071,0.039872408,
    0.265816055,0,0.021265284,0.318979266,0.070707071],
7: [0.11, 0.21, 0.06, 0.17, 0.08, 0.06, 0.05, 0, 0.09, 0.17],
8: [0.25, 0.02, 0.1, 0.1, 0.05, 0.1, 0.25, 0.03, 0, 0.1],
9: [0.08, 0.04, 0.05, 0.12, 0.26, 0.12, 0.08, 0.09, 0.16, 0]
}


probabilities_0 = [0, 0.03, 0.08, 0.08, 0.05, 0.08, 0.16, 0.06, 0.38, 0.08]
probabilities_1 = [0.09, 0, 0.03, 0.12, 0.17, 0.03, 0.03, 0.34, 0.07, 0.12]
probabilities_2 = [0.14, 0.02, 0, 0.2, 0.04, 0.08, 0.14, 0.05, 0.25, 0.08]
probabilities_3 = [0.09, 0.05, 0.14, 0, 0.07, 0.14, 0.09, 0.1, 0.18, 0.14]
probabilities_4 = [0.07, 0.09, 0.04, 0.09, 0, 0.09, 0.07, 0.06, 0.12, 0.37]
probabilities_5 = [0.09, 0.01, 0.05, 0.13, 0.07, 0, 0.33, 0.03, 0.16, 0.13]
probabilities_6 = [0.132908028,0.009037746,0.070707071,0.070707071,0.039872408,
                   0.265816055,0,0.021265284,0.318979266,0.070707071]
probabilities_7 = [0.11, 0.21, 0.06, 0.17, 0.08, 0.06, 0.05, 0, 0.09, 0.17]
probabilities_8 = [0.25, 0.02, 0.1, 0.1, 0.05, 0.1, 0.25, 0.03, 0, 0.1]
probabilities_9 = [0.08, 0.04, 0.05, 0.12, 0.26, 0.12, 0.08, 0.09, 0.16, 0]

def assign_labels(dataset, output):
    # Initialize an empty list to store generated labels
    generated_labels1 = []
    generated_labels2 = []
    generated_labels3 = []
    generated_labels4 = []

    # Generate random labels based on probabilities for each number in the original column
    for original_label in dataset['Label']:

        probabilities = probabilities_lists[original_label]
        generated_label1 = np.random.choice(range(10), p=probabilities)
        generated_label2 = np.random.choice(range(10), p=probabilities)
        generated_label3 = np.random.choice(range(10), p=probabilities)
        generated_label4 = np.random.choice(range(10), p=probabilities)

        # To ensure labels within each row are unique
        while generated_label3 == generated_label4 or generated_label2 == generated_label4 or generated_label3 == generated_label2 or generated_label1 == generated_label4 or generated_label1 == generated_label3 or generated_label1 == generated_label2:
            generated_label1 = np.random.choice(range(10), p=probabilities)
            generated_label2 = np.random.choice(range(10), p=probabilities)
            generated_label3 = np.random.choice(range(10), p=probabilities)
            generated_label4 = np.random.choice(range(10), p=probabilities)
        generated_labels1.append(generated_label1)
        generated_labels2.append(generated_label2)
        generated_labels3.append(generated_label3)
        generated_labels4.append(generated_label4)


    # Add the generated labels as a new column to the DataFrame
    dataset['Generated_Labels1'] = generated_labels1
    dataset['Generated_Labels2'] = generated_labels2
    dataset['Generated_Labels3'] = generated_labels3
    dataset['Generated_Labels4'] = generated_labels4

    agreement_path = os.path.join(folder_path, 'five_raters' , output)
    dataset.to_csv(agreement_path, index=False)

assign_labels(agreement_0_test_data, 'agreement_0_test.csv')

def pivot_for_model(dataset, instance_id, output):
    instance_counter = instance_id
    dataset.sort_values(by='image_id', inplace=True)

    results_df = pd.DataFrame()
    for label in ['Label', 'Generated_Labels1', 'Generated_Labels2', 'Generated_Labels3', 'Generated_Labels4']:
        for index, row in dataset.iterrows():
            new_row = pd.DataFrame({
                'image_id': [row['image_id']],
                'instance_id': [instance_counter],  # Assign the current counter value
                'flattened_image': [row['flattened_image']],
                'agreement': [row['agreement']],
                'Label': [row[label]]
            })
            results_df = pd.concat([results_df, new_row], ignore_index=True)
            instance_counter += 1

    output_path = os.path.join(folder_path, 'five_raters/model_ready/', output)
    results_df.to_csv(output_path, index=False)


pivot_for_model(agreement_0_test_data, 282001, 'agreement_0_test.csv')