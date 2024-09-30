###Sharpening Filters
# In[4]: Using Laplacian Kernal


import cv2
import matplotlib.pyplot as plt
import numpy as np
image1=cv2.imread("red.jpg")
image2=cv2.cvtColor(image1,cv2.COLOR_BGR2RGB)
kernel=np.ones((11,11),np.float32)/121
image3=cv2.filter2D(image2,-1,kernel)
kernel2 = np.array([[0,1,0], [1, -4,1],[0,1,0]])
image3=cv2.filter2D(image2,, -1, kernel2)
plt.figure(figsize=(10,8))
plt.subplot(1,2,2)
plt.imshow(image2)
plt.title("Original Image")
plt.axis("off")
plt.subplot(1,2,2)
plt.title("Using Laplacian Kernal")
plt.imshow(image3)
plt.axis("off")



# In[5]:Using Laplacian Operator


laplacian = cv2.Laplacian(image2, cv2.CV_64F)
plt.figure(figsize=(10, 8))
plt.subplot(1, 2, 2)
plt.imshow(laplacian, cmap='gray')
plt.title("Laplacian Operator Image")
plt.axis("off")
plt.show()





