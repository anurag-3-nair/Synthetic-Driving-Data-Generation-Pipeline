import cv2
import matplotlib.pyplot as plt

img = cv2.imread("dataset/img_0.png")
edges = cv2.Canny(img, 100, 200)

plt.imshow(edges, cmap="gray")
plt.title("Edge Detection")
plt.show()