import cv2
import numpy as np

# Load the image
image_path_corrected = 'percep.png'
image = cv2.imread(image_path_corrected)

# Convert to grayscale
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Apply Gaussian blur to reduce noise
blurred_image = cv2.GaussianBlur(gray_image, (5, 5), 0)

# Use Canny edge detection or adaptive thresholding to highlight edges
edges = cv2.Canny(blurred_image, 50, 150)

# Find contours in the edge-detected image
contours, _ = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

# Initialize lists for left and right cone centroids
left_cone_centroids = []
right_cone_centroids = []

# Loop over contours to detect and classify cones
for cnt in contours:
    area = cv2.contourArea(cnt)
    if area > 100:  # Filter small areas
        M = cv2.moments(cnt)
        if M["m00"] != 0:
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])
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

# Save the output image with corrected parallel lines
output_image_parallel_lines_path = 'answer_grayscale.png'
cv2.imwrite(output_image_parallel_lines_path, image)

print("Processing complete. The output image has been saved.")
