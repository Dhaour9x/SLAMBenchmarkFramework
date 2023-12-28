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

        for point in aligned_points:
            x, y = point
            cv2.circle(result_image, (int(x), int(y)), 5, (255, 0, 0), -1)

        for point in reference_points:
            x, y = point
            cv2.circle(result_image, (int(x), int(y)), 5, (0, 255, 0), -1)

        for point in target_points:
            x, y = point
            cv2.circle(result_image, (int(x), int(y)), 5, (0, 0, 255), -1)

        cv2.putText(result_image, f"Iteration: {k + 1}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2,
                    cv2.LINE_AA)
        cv2.putText(result_image, f"ICP Error: {cost:.4f}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2,
                    cv2.LINE_AA)
        cv2.putText(result_image, "Reference Corners", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2,
                    cv2.LINE_AA)
        cv2.putText(result_image, "Target Corners", (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2,
                    cv2.LINE_AA)
        cv2.putText(result_image, "Matched Corners", (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2,
                    cv2.LINE_AA)

        # Display the final alignment
        cv2.imshow("ICP Progress", result_image)
        cv2.waitKey(100)  # Adjust the delay between iterations (ms)

    return R, t, errors


# Load your images
image1 = cv2.imread("Zang_thining_GroundTruthClean.png")
image2 = cv2.imread("Zang_thining_CartographerClean.png")

# Convert images to grayscale
gray1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
gray2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)

# Detect corners in the images
corners1 = cv2.goodFeaturesToTrack(gray1, maxCorners=100, qualityLevel=0.01, minDistance=10)
corners2 = cv2.goodFeaturesToTrack(gray2, maxCorners=100, qualityLevel=0.01, minDistance=10)

# Convert corners to numpy arrays
corners1 = np.intp(corners1)
corners2 = np.intp(corners2)

# Extract (x, y) coordinates of corners
reference_points = [tuple(c[0]) for c in corners1]
target_points = [tuple(c[0]) for c in corners2]

# Perform ICP
R, t, errors = icp(np.array(reference_points), np.array(target_points))

# R and t represent the transformation from reference_points to target_points
print("Rotation matrix R:\n", R)
print("Translation vector t:\n", t)

# Display the final alignment with title and legend
aligned_points = np.dot(R, np.array(reference_points).T).T + t
result_image = np.zeros_like(image2)
result_image.fill(255)

for point in aligned_points:
    x, y = point
    cv2.circle(result_image, (int(x), int(y)), 5, (255, 0, 0), -1)

for point in reference_points:
    x, y = point
    cv2.circle(result_image, (int(x), int(y)), 5, (0, 255, 0), -1)

for point in target_points:
    x, y = point
    cv2.circle(result_image, (int(x), int(y)), 5, (0, 0, 255), -1)

cv2.putText(result_image, "Final Alignment", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2, cv2.LINE_AA)
cv2.putText(result_image, f"Mean ICPc Error: {np.mean(errors):.4f}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0),
            2, cv2.LINE_AA)
cv2.putText(result_image, "Reference Corners", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_AA)
cv2.putText(result_image, "Target Corners", (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2, cv2.LINE_AA)
cv2.putText(result_image, "Matched Corners", (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2, cv2.LINE_AA)

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
