from data_folder_path import folder_path

import pydicom

dicom_file = pydicom.dcmread('/Users/joannelin/Data/LIDC/crops/1.dcm')



# Get the pixel array shape
pixel_array_shape = dicom_file.pixel_array.shape

print(f"The shape of the image is: {pixel_array_shape}")
