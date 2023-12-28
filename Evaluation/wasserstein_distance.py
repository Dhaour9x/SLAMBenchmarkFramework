import numpy as np
from scipy.stats import wasserstein_distance
import matplotlib.pyplot as plt
from skimage.io import imread

'''The Wasserstein distance, also known as the Earth Mover's distance, is a measure of the distance between two 
probability distributions. It calculates the minimum amount of work required to transform one distribution into the 
other. In the context of images, the Wasserstein distance measures the minimum amount of "mass" that needs to be 
moved to transform one image into another. In the context of image analysis, the term "probability distribution" 
refers to the statistical distribution of pixel intensities within an image. Each pixel in an image has an associated 
intensity value, which can range from 0 to a maximum value (e.g., 255 for an 8-bit grayscale image). For a grayscale 
image, the probability distribution is typically represented as a histogram. The histogram shows the number of pixels 
in the image that have a specific intensity value. The x-axis of the histogram represents the intensity values, 
and the y-axis represents the frequency or probability of occurrence. 

By computing the Wasserstein distance between two images, we can quantify the dissimilarity or similarity between 
them. A lower Wasserstein distance indicates a higher similarity, while a higher distance indicates a greater 
dissimilarity. It's important to note that the Wasserstein distance is a metric that considers the spatial 
relationship between pixel values in the images. It provides a more robust measure of dissimilarity compared to 
simple pixel-wise comparisons, as it takes into account the underlying structure and distribution of the image data. '''
# Load images
image1 = imread('GroundTruthClean.png')
image2 = imread('CartographerClean.png')

# Calculate the Wasserstein distance
w_distance = wasserstein_distance(image1.flatten(), image2.flatten())

# Create a single plot
fig, ax = plt.subplots()

# Show image 1 in red
ax.imshow(image1, cmap='Reds', alpha=0.5)

# Show image 2 in green
ax.imshow(image2, cmap='Greens', alpha=0.5)

# Set plot title and axis labels
ax.set_title('Wasserstein Distance: {:.2f}'.format(w_distance))
ax.axis('off')

# Display the plot
plt.show()
