import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load the image uploaded by the user
image_path = 'percep.png'
image = cv2.imread(image_path)

# Convert to HSV color space for better color segmentation
hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

# Define the red color range for detecting the cones
lower_red = np.array([174, 28, 32])
upper_red = np.array([254, 75, 78])
mask = cv2.inRange(hsv_image, lower_red, upper_red)

# Find contours in the red mask
contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

# Initialize lists for left and right cone centroids
left_cone_centroids = []
right_cone_centroids = []

# Loop over contours to detect and classify cones
for cnt in contours:
    area = cv2.contourArea(cnt)
    if area > 100:  # Filter small areas
        M = cv2.moments(cnt)
        if M["m00"] != 0:
            cX = int(M["m10"] / M["m00"])  # Centroid X
            cY = int(M["m01"] / M["m00"])  # Centroid Y
            # Separate left and right cones based on x-coordinate
            if cX < image.shape[1] // 2:
                left_cone_centroids.append((cX, cY))
            else:
                right_cone_centroids.append((cX, cY))

# Sort centroids by Y coordinate to ensure proper line drawing
left_cone_centroids = sorted(left_cone_centroids, key=lambda x: x[1])
right_cone_centroids = sorted(right_cone_centroids, key=lambda x: x[1])

# Draw the left red line over the left cones
for i in range(len(left_cone_centroids) - 1):
    cv2.line(image, left_cone_centroids[i], left_cone_centroids[i + 1], (0, 0, 255), 5)

# Draw the right red line over the right cones
for i in range(len(right_cone_centroids) - 1):
    cv2.line(image, right_cone_centroids[i], right_cone_centroids[i + 1], (0, 0, 255), 5)

# Save the output image with lines drawn
output_image_path = 'answer_with_lines.png'
cv2.imwrite(output_image_path, image)

# Display the result
output_image_path
