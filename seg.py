import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.segmentation import watershed
from skimage.feature import peak_local_max
from skimage import color

# Đọc ảnh và chuyển đổi thành ảnh xám
image = cv2.imread('path_to_your_image.jpg')
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Làm mượt ảnh
blurred_image = cv2.GaussianBlur(gray_image, (5, 5), 0)

# Tính toán gradient của ảnh
gradient = cv2.morphologyEx(blurred_image, cv2.MORPH_GRADIENT, cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)))

# Tạo ảnh marker cho thuật toán Watershed
# 0 là vùng nền, các giá trị khác là các marker cho các đối tượng
markers = np.zeros_like(gray_image)
markers[gray_image < 50] = 1  # Ví dụ: các điểm sáng (nền) trong ảnh có thể có giá trị thấp hơn 50
markers[gray_image > 150] = 2  # Ví dụ: các điểm tối (đối tượng) trong ảnh có thể có giá trị cao hơn 150

# Áp dụng thuật toán Watershed
# Chuyển đổi gradient từ ảnh xám thành ảnh nhị phân để sử dụng trong thuật toán Watershed
distance_transform = cv2.distanceTransform(gradient.astype(np.uint8), cv2.DIST_L2, 5)
# Tạo ảnh marker từ distance_transform
local_max = peak_local_max(distance_transform, indices=False, footprint=np.ones((3, 3)), labels=markers)
markers[local_max] = 3

# Áp dụng Watershed
labels = watershed(-distance_transform, markers, mask=gradient)

# Hiển thị kết quả phân đoạn
plt.figure(figsize=(12, 8))
plt.subplot(1, 3, 1)
plt.title('Original Image')
plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
plt.subplot(1, 3, 2)
plt.title('Gradient Image')
plt.imshow(gradient, cmap='gray')
plt.subplot(1, 3, 3)
plt.title('Watershed Result')
plt.imshow(labels, cmap='jet')
plt.show()
