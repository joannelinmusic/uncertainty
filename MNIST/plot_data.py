from read_data import full_test_data, full_train_data, full_validation_data, high_test_data, high_validation_data, high_train_data, low_validation_data, low_test_data, low_train_data, no_test_data, no_train_data, no_validation_data

import matplotlib.pyplot as plt
import numpy as np

# Pick an image (pixels)
image_pixels = full_train_data['flattened_image'][1]

# Remove non-numeric characters 
# convert the string to numbers 
image_pixels = np.array(image_pixels.replace('[', '').replace(']', '').split(','), dtype=int)

# Reshape the flattened image data to 28x28 
image_pixels = image_pixels.reshape(28, 28)

# Plotting the image
plt.imshow(image_pixels, cmap='gray')
plt.colorbar()
plt.title('28x28 Pixel Image')
plt.show()
