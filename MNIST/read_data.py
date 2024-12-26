from data_folder_path import folder_path
import pandas as pd

# amal
full_test_data = pd.read_csv(folder_path + 'Amal_dataset/' + 'full_test_data.csv')
full_train_data = pd.read_csv(folder_path + 'Amal_dataset/' + 'full_train_data.csv')
full_validation_data = pd.read_csv(folder_path + 'Amal_dataset/' + 'full_validation_data.csv')
high_test_data = pd.read_csv(folder_path + 'Amal_dataset/' + 'high_test_data.csv')
high_train_data = pd.read_csv(folder_path + 'Amal_dataset/' + 'high_train_data.csv')
high_validation_data = pd.read_csv(folder_path + 'Amal_dataset/' + 'high_validation_data.csv')
low_test_data = pd.read_csv(folder_path + 'Amal_dataset/' + 'low_test_data.csv')
low_train_data = pd.read_csv(folder_path + 'Amal_dataset/' + 'low_train_data.csv')
low_validation_data = pd.read_csv(folder_path + 'Amal_dataset/' + 'low_validation_data.csv')
no_test_data = pd.read_csv(folder_path + 'Amal_dataset/' + 'no_test_data.csv')
no_train_data = pd.read_csv(folder_path + 'Amal_dataset/' + 'no_train_data.csv')
no_validation_data = pd.read_csv(folder_path + 'Amal_dataset/' + 'no_validation_data.csv')

# five raters - model not ready
agreement_0_train_data = pd.read_csv(folder_path + 'five_raters/' + 'agreement_0_train.csv')
agreement_1_train_data = pd.read_csv(folder_path + 'five_raters/' + 'agreement_1_train.csv')
agreement_2_train_data = pd.read_csv(folder_path + 'five_raters/' + 'agreement_2_train.csv')
agreement_3_train_data = pd.read_csv(folder_path + 'five_raters/' + 'agreement_3_train.csv')
agreement_4_train_data = pd.read_csv(folder_path + 'five_raters/' + 'agreement_4_train.csv')
agreement_0_test_data = pd.read_csv(folder_path + 'five_raters/' + 'agreement_0_test.csv')
agreement_1_test_data = pd.read_csv(folder_path + 'five_raters/' + 'agreement_1_test.csv')
agreement_2_test_data = pd.read_csv(folder_path + 'five_raters/' + 'agreement_2_test.csv')
agreement_3_test_data = pd.read_csv(folder_path + 'five_raters/' + 'agreement_3_test.csv')
agreement_4_test_data = pd.read_csv(folder_path + 'five_raters/' + 'agreement_4_test.csv')


# five raters - model ready
agreement_0_train = pd.read_csv(folder_path + 'five_raters/model_ready/' + 'agreement_0_train.csv')
agreement_1_train = pd.read_csv(folder_path + 'five_raters/model_ready/' + 'agreement_1_train.csv')
agreement_2_train = pd.read_csv(folder_path + 'five_raters/model_ready/' + 'agreement_2_train.csv')
agreement_3_train = pd.read_csv(folder_path + 'five_raters/model_ready/' + 'agreement_3_train.csv')
agreement_4_train = pd.read_csv(folder_path + 'five_raters/model_ready/' + 'agreement_4_train.csv')
agreement_0_test = pd.read_csv(folder_path + 'five_raters/model_ready/' + 'agreement_0_test.csv')
agreement_1_test = pd.read_csv(folder_path + 'five_raters/model_ready/' + 'agreement_1_test.csv')
agreement_2_test = pd.read_csv(folder_path + 'five_raters/model_ready/' + 'agreement_2_test.csv')
agreement_3_test = pd.read_csv(folder_path + 'five_raters/model_ready/' + 'agreement_3_test.csv')
agreement_4_test = pd.read_csv(folder_path + 'five_raters/model_ready/' + 'agreement_4_test.csv')