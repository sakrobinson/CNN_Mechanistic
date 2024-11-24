import numpy as np
import cv2
import matplotlib.pyplot as plt

def create_synthetic_dataset(num_samples=1000, img_size=64):
    images = []
    labels = []
    for _ in range(num_samples):
        # Create a blank image
        img = np.zeros((img_size, img_size, 3), dtype=np.uint8)
        # Randomly choose a shape
        shape_type = np.random.choice(['circle', 'square'])
        color = (255, 255, 255)  # White color
        if shape_type == 'circle':
            center = (np.random.randint(15, img_size - 15), np.random.randint(15, img_size - 15))
            radius = np.random.randint(5, 15)
            cv2.circle(img, center, radius, color, -1)
            labels.append(0)  # Label for circle
        else:
            top_left = (np.random.randint(0, img_size - 30), np.random.randint(0, img_size - 30))
            bottom_right = (top_left[0] + np.random.randint(15, 30), top_left[1] + np.random.randint(15, 30))
            cv2.rectangle(img, top_left, bottom_right, color, -1)
            labels.append(1)  # Label for square
        images.append(img)
    return np.array(images), np.array(labels)

# Generate the dataset
X, y = create_synthetic_dataset(num_samples=2000)

# Visualize some samples
fig, axes = plt.subplots(1, 5, figsize=(15, 3))
for i, ax in enumerate(axes.flat):
    ax.imshow(X[i])
    ax.set_title('Circle' if y[i] == 0 else 'Square')
    ax.axis('off')
plt.show()
