# Basic Libraries
import time
import pandas as pd
import os
import numpy as np
import pydicom
from skimage.transform import resize

# ML libraries
from sklearn import svm, utils
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score

# Load libraries
from joblib import dump, load
from data_folder_path import folder_path

# Data
from read_data import full_test_data, full_train_data, high_test_data, high_train_data, low_test_data, low_train_data, no_test_data, no_train_data


start_time = time.time()

def concatenate_datasets(*datasets):
    return pd.concat(datasets, ignore_index=True)

def process_image_data(dataframe, image_column, target_shape=(28, 28)):
    """
    Processes a column of flattened images stored as strings in a dataframe,
    converting them to a numpy array of integers.

    Returns:
    np.array: A numpy array where each row is an image represented as integers.
    """
    # Convert each flattened image from a string to a numpy array of integers
    images = []
    for file_path in dataframe[image_column]:
        # Load the DICOM file
        dicom_file = pydicom.dcmread(file_path)
        pixel_array = dicom_file.pixel_array
        
        # Resize the image to the target shape
        resized_image = resize(pixel_array, target_shape, anti_aliasing=True)
        
        # Flatten the image and append it
        images.append(resized_image.flatten().astype(float))  # Ensure uniform type
    
    # Convert to numpy array
    X = np.array(images)
    return X

def load_model(model_path):
    if os.path.exists(model_path):
        model = load(model_path)
        return model
    else:
        print("Model file not found.")
        return None

def train_SVC(train, model_path):

    data_shuffle = utils.shuffle(train)

    X_train = process_image_data(data_shuffle, 'image')
    y_train = data_shuffle['label'].values

    # train SVM  
    model = svm.SVC()
    model.fit(X_train, y_train)

    # save model
    dump(model, model_path)
    return model

def train_RFC(train, model_path):

    data_shuffle = utils.shuffle(train)

    X_train = process_image_data(data_shuffle, 'image')
    y_train = data_shuffle['label'].values

    # train SVM  
    model = RandomForestClassifier(n_estimators=100)
    model.fit(X_train, y_train)

    # save model
    dump(model, model_path)
    return model


def testing(test, model):
    data_shuffle = utils.shuffle(test)

    X_test = process_image_data(data_shuffle, 'image')
    y_test = data_shuffle['label'].values
    # Predict
    y_pred = model.predict(X_test)

    # Evaluate
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("\nClassification Report:\n", classification_report(y_test, y_pred))

combined_train_data = concatenate_datasets(low_train_data, no_train_data, high_train_data, full_train_data)

model_path = folder_path+'models/one_model_2.joblib'

if not os.path.exists(model_path):
    print('Model does not previously exist. Training now.')
    # model = train_SVC(combined_train_data, model_path)
    model = train_RFC(combined_train_data, model_path)
else:
    print('Model exists. Testing starts now.')
    model = load_model(model_path)

testing(full_test_data, model)
testing(high_test_data, model)
testing(low_test_data, model)
testing(no_test_data, model)


end_time = time.time()

# Calculate the runtime
runtime = end_time - start_time

# Print the runtime
print(f"The runtime of the code is {runtime} seconds")