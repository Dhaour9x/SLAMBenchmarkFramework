import cv2
import numpy as np
import matplotlib.pyplot as plt

# Define the ICP function
def icp(reference_points, target_points, max_iterations=24, tolerance=1e-6):
    R = np.identity(2)  # Initialize rotation matrix
    t = np.zeros(2)  # Initialize translation vector

    cv2.namedWindow("ICP Progress", cv2.WINDOW_NORMAL)

    errors = []  # List to store mean ICP error for each iteration

    for k in range(max_iterations):
        # Step 2: Compute the closest point correspondence
        transformed_reference = np.dot(R, reference_points.T).T + t
        correspondences = []
        for i in range(transformed_reference.shape[0]):
            distances = np.linalg.norm(target_points - transformed_reference[i], axis=1)
            min_distance_idx = np.argmin(distances)
            correspondences.append((i, min_distance_idx))

        # Extract corresponding points
        corresponding_reference = np.array([reference_points[i] for i, _ in correspondences])
        corresponding_target = np.array([target_points[j] for _, j in correspondences])

        # Step 3: Compute the registration
        mean_reference = np.mean(corresponding_reference, axis=0)
        mean_target = np.mean(corresponding_target, axis=0)

        centered_reference = corresponding_reference - mean_reference
        centered_target = corresponding_target - mean_target

        W = np.dot(centered_target.T, centered_reference)
        U, _, Vt = np.linalg.svd(W)
        R_new = np.dot(U, Vt)
        t_new = mean_target - np.dot(R_new, mean_reference)

        # Step 4: Check for convergence
        cost = np.mean(
            np.linalg.norm(corresponding_target - np.dot(R_new, corresponding_reference.T).T - t_new, axis=1))
        errors.append(cost)

        if abs(cost) < tolerance:
            break

        R = np.dot(R_new, R)
        t = t_new

        # Update target points for the next iteration
        target_points = np.dot(R, np.array(reference_points).T).T + t

        # Visualize the alignment (update after each iteration)
        aligned_points = np.dot(R, np.array(reference_points).T).T + t
        result_image = np.zeros_like(image2)
        result_image.fill(255)

        # Draw reference edges in green
        for i in range(len(reference_points) - 1):
            pt1 = tuple(map(int, reference_points[i]))
            pt2 = tuple(map(int, reference_points[i + 1]))
            cv2.line(result_image, pt1, pt2, (0, 255, 0), 2)

        # Draw target edges in red
        for i in range(len(target_points) - 1):
            pt1 = tuple(map(int, target_points[i]))
            pt2 = tuple(map(int, target_points[i + 1]))
            cv2.line(result_image, pt1, pt2, (255, 0, 0), 2)

        # Draw matched edges in blue
        for i in range(len(aligned_points) - 1):
            pt1 = tuple(map(int, aligned_points[i]))
            pt2 = tuple(map(int, aligned_points[i + 1]))
            cv2.line(result_image, pt1, pt2, (0, 0, 255), 2)

        cv2.putText(result_image, f"Iteration: {k + 1}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2,
                    cv2.LINE_AA)
        cv2.putText(result_image, f"ICP Error: {cost:.4f}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2,
                    cv2.LINE_AA)
        cv2.putText(result_image, "Reference Edges", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2,
                    cv2.LINE_AA)
        cv2.putText(result_image, "Target Edges", (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2,
                    cv2.LINE_AA)
        cv2.putText(result_image, "Matched Edges", (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2,
                    cv2.LINE_AA)

        # Display the final alignment
        cv2.imshow("ICP Progress", result_image)
        cv2.waitKey(100)  # Adjust the delay between iterations (ms)

    return R, t, errors


# Load your images
image1 = cv2.imread("Zang_thining_CartographerClean.png")
image2 = cv2.imread("Zang_thining_HectorClean.png")

# Convert images to grayscale
gray1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
gray2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)

# Apply edge detection to obtain edges
edges1 = cv2.Canny(gray1, 50, 150)
edges2 = cv2.Canny(gray2, 50, 150)

# Handle different OpenCV versions
if cv2.__version__.startswith('4'):
    contours1, _ = cv2.findContours(edges1, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours2, _ = cv2.findContours(edges2, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
else:
    _, contours1, _ = cv2.findContours(edges1, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    _, contours2, _ = cv2.findContours(edges2, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Extract (x, y) coordinates of contour points
reference_points = [tuple(point[0]) for contour in contours1 for point in contour]
target_points = [tuple(point[0]) for contour in contours2 for point in contour]

# Perform ICP
R, t, errors = icp(np.array(reference_points), np.array(target_points))

# Display the final alignment with title and legend
aligned_points = np.dot(R, np.array(reference_points).T).T + t
result_image = np.zeros_like(image2)
result_image.fill(255)

# Draw reference edges in green
for i in range(len(reference_points) - 1):
    pt1 = tuple(map(int, reference_points[i]))
    pt2 = tuple(map(int, reference_points[i + 1]))
    cv2.line(result_image, pt1, pt2, (0, 255, 0), 2)

# Draw target edges in red
for i in range(len(target_points) - 1):
    pt1 = tuple(map(int, target_points[i]))
    pt2 = tuple(map(int, target_points[i + 1]))
    cv2.line(result_image, pt1, pt2, (255, 0, 0), 2)

# Draw matched edges in blue
for i in range(len(aligned_points) - 1):
    pt1 = tuple(map(int, aligned_points[i]))
    pt2 = tuple(map(int, aligned_points[i + 1]))
    cv2.line(result_image, pt1, pt2, (0, 0, 255), 2)

cv2.putText(result_image, "Final Alignment", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2, cv2.LINE_AA)
cv2.putText(result_image, f"Mean ICPe Error: {np.mean(errors):.4f}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0),
            2, cv2.LINE_AA)
cv2.putText(result_image, "Reference Edges", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_AA)
cv2.putText(result_image, "Target Edges", (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2, cv2.LINE_AA)
cv2.putText(result_image, "Matched Edges", (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2, cv2.LINE_AA)

# Display the final alignment
cv2.imshow("Final Alignment", result_image)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Plot the ICP error for each iteration
# plt.plot(errors, marker='o')
# plt.title('ICP Error Over Iterations')
# plt.xlabel('Iteration')
# plt.ylabel('ICP Error')
# plt.show()
