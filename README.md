# Ex.No 5: Implementation-of-filter
## Aim:
To implement filters for smoothing and sharpening the images in the spatial domain.

## Software Required:
Anaconda - Python 3.7

## Algorithm:
### Step 1: 
Import necessary libraries (cv2, NumPy, Matplotlib) for image loading, filtering, and visualization.

### Step 2: 
Load the image using cv2.imread() and convert it to RGB format using cv2.cvtColor() for proper display in Matplotlib.

### Step 3: 
Apply different filters:
1. Averaging Filter: Define an averaging kernel using np.ones() and apply it to the image using cv2.filter2D().
2. Weighted Averaging Filter: Define a weighted kernel (e.g., 3x3 Gaussian-like) and apply it with cv2.filter2D().
3. Gaussian Filter: Use cv2.GaussianBlur() to apply Gaussian blur.
4. Median Filter: Use cv2.medianBlur() to reduce noise.
5. Laplacian Operator: Use cv2.Laplacian() to apply edge detection.
    
### Step 4: 
Display each filtered image using plt.subplot() and plt.imshow() for side-by-side comparison of the original and processed images.

### Step 5: 
Save or show the images using plt.show() after applying each filter to visualize the effects of smoothing and sharpening.


## Program:
### Developed By   : SARAVANA KUMAR M
### Register Number: 212222230133
</br>

### 1. Smoothing Filters

#### Original Image
```
import cv2
import matplotlib.pyplot as plt
import numpy as np
image1 = cv2.imread("IMAGE2.webp")
image2 = cv2.cvtColor(image1, cv2.COLOR_BGR2RGB)
plt.figure(figsize=(10, 8))
plt.subplot(1, 2, 1)
plt.imshow(image2)
plt.title("Original Image")
plt.axis("off")

plt.show()
```
#### i) Using Averaging Filter
```Python
kernel = np.ones((11, 11), np.float32) / 121
averaging_image = cv2.filter2D(image2, -1, kernel)
plt.figure(figsize=(10, 8))
plt.subplot(1, 2, 2)
plt.imshow(averaging_image)
plt.title("Averaging Filter Image")
plt.axis("off")
plt.show()
```
#### ii) Using Weighted Averaging Filter
```Python
kernel1 = np.array([[1, 2, 1],
                    [2, 4, 2],
                    [1, 2, 1]]) / 16

weighted_average_image = cv2.filter2D(image2, -1, kernel1)
plt.figure(figsize=(10, 8))
plt.subplot(1, 2, 2)
plt.imshow(weighted_average_image)
plt.title("Weighted Average Filter Image")
plt.axis("off")
plt.show()

```
#### iii) Using gaussian Filter
```Python
gaussian_blur = cv2.GaussianBlur(image2, (11, 11), 0)
plt.figure(figsize=(10, 8))
plt.subplot(1, 2, 2)
plt.imshow(gaussian_blur)
plt.title("Gaussian Blur")
plt.axis("off")
plt.show()
```

#### iv) Using Median Filter
```Python
median_blur = cv2.medianBlur(image2, 11)
plt.figure(figsize=(10, 8))
plt.subplot(1, 2, 2)
plt.imshow(median_blur)
plt.title("Median Filter")
plt.axis("off")
plt.show()
```

### 2. Sharpening Filters
#### i) Using Laplacian Linear Kernal
```Python
image1 = cv2.imread('Ex_5_image.jpeg')
image2 = cv2.cvtColor(image1, cv2.COLOR_BGR2RGB)

kernel3 = np.array([[0,1,0], [1, -4,1],[0,1,0]])
image5 =cv2.filter2D(image2, -1, kernel3)
plt.imshow(image5)
plt.title('Laplacian Kernel')
```
#### ii) Using Laplacian Operator
```Python
laplacian = cv2.Laplacian(image2, cv2.CV_64F)
plt.figure(figsize=(10, 8))
plt.subplot(1, 2, 2)
plt.imshow(laplacian, cmap='gray')
plt.title("Laplacian Operator Image")
plt.axis("off")
plt.show()
```

## OUTPUT:
#### Original Image
![image](https://github.com/user-attachments/assets/17c819d1-48c2-4726-b49b-6a34ef016761)

### 1. Smoothing Filters
#### i) using Averaging Filter Image
![image](https://github.com/user-attachments/assets/6ec2221c-ae1a-4d7e-bf70-116f19857608)

#### ii) Using Weighted Average Filter Image
![image](https://github.com/user-attachments/assets/9fc953df-69c2-4a19-be2b-51a3a5473765)


#### iii) Using Gaussian Filter
![image](https://github.com/user-attachments/assets/9833e556-9771-42f8-bdce-08c1600ad03a)


#### iv) Using Median Filter
![image](https://github.com/user-attachments/assets/51564fa3-51eb-4d0b-a8f5-ef4c36c4b23e)


### 2. Sharpening Filters
#### i) Using Laplacian Kernal
![image](https://github.com/user-attachments/assets/817ec612-627a-4238-a2ef-f2b8962b9ed3)


#### ii) Using Laplacian Operator
![image](https://github.com/user-attachments/assets/cb1884c2-117e-4665-bc97-db01a92df698)


## Result:
Thus the filters are designed for smoothing and sharpening the images in the spatial domain.
