import cv2
import numpy as np

# Loading the image
image_path_corrected = 'percep.png'
image = cv2.imread(image_path_corrected)

# Converting the image HSV color space for better color segmentation
hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

# Defining the red color range for detecting the cones
lower_red_1 = np.array([0, 170, 150])  # Further increased saturation and value to filter out the door and other objects
upper_red_1 = np.array([10, 255, 255])  # Upper bound 
mask1 = cv2.inRange(hsv_image, lower_red_1, upper_red_1)

lower_red_2 = np.array([170, 150, 130])  # 2nd pair of bounds to increase accuracy, and ensure we detect all the hue in the red
upper_red_2 = np.array([180, 255, 255])
mask2 = cv2.inRange(hsv_image, lower_red_2, upper_red_2)

# this combines both masks for red color detection
mask = mask1 | mask2

# finding the outlines in the red mask
contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

# this initializes the lists for the left and right cone centroids
left_cone_centroids = []
right_cone_centroids = []

# Here we loop over the contours(outlines) to detect and classify cones
for colorred in contours:
    area = cv2.contourArea(colorred)
    if area > 100:  # Filtering out the small areas
        M = cv2.moments(colorred)
        if M["m00"] != 0:
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])
            # Here we are separating left and right cones based on x-coordinate
            if cX < image.shape[1] // 2:
                left_cone_centroids.append((cX, cY))
            else:
                right_cone_centroids.append((cX, cY))

# Sorting the centroids by Y coordinate so the line can draw properly
left_cone_centroids = sorted(left_cone_centroids, key=lambda x: x[1])
right_cone_centroids = sorted(right_cone_centroids, key=lambda x: x[1])

# Drawing the left red line over the left cones
for i in range(len(left_cone_centroids) - 1):
    cv2.line(image, left_cone_centroids[i], left_cone_centroids[i + 1], (0, 0, 255), 5)

# Drawing the right red line over the right cones
for i in range(len(right_cone_centroids) - 1):
    cv2.line(image, right_cone_centroids[i], right_cone_centroids[i + 1], (0, 0, 255), 5)

# Saveing and writing the output image with corrected lines
output = 'answer.png'
cv2.imwrite(output, image)

print("Processing complete. The output image has been saved.")
