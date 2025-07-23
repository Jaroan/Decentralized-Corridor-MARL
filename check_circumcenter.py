import numpy as np
def compute_circumcenter(p1, p2, p3):
    """
    Computes the circumcenter of a triangle formed by three points.

    Parameters:
        p1, p2, p3: Tuples or lists representing the (x, y) coordinates of the three points.

    Returns:
        (Xc, Yc): The circumcenter coordinates.
    """
    x1, y1 = p1
    x2, y2 = p2
    x3, y3 = p3

    # Compute determinant
    D = 2 * (x1 * (y2 - y3) + x2 * (y3 - y1) + x3 * (y1 - y2))
    
    # Avoid division by zero in degenerate cases (collinear points)
    if abs(D) < 1e-6:
        return None  # Circumcenter is undefined (or at infinity)

    # Compute circumcenter coordinates
    Xc = ((x1**2 + y1**2) * (y2 - y3) + (x2**2 + y2**2) * (y3 - y1) + (x3**2 + y3**2) * (y1 - y2)) / D
    Yc = ((x1**2 + y1**2) * (x3 - x2) + (x2**2 + y2**2) * (x1 - x3) + (x3**2 + y3**2) * (x2 - x1)) / D

    return (Xc, Yc)

def reward_function(circumcenter, target_center):
    """
    Computes the reward based on how close the computed circumcenter is to the target center.

    Parameters:
        circumcenter: Tuple (Xc, Yc), the computed circumcenter.
        target_center: Tuple (Xt, Yt), the actual center of the desired circle formation.

    Returns:
        Reward value (negative Euclidean distance).
    """
    if circumcenter is None:
        return -100  # Large penalty for degenerate cases

    Xc, Yc = circumcenter
    Xt, Yt = target_center

    # Euclidean distance penalty
    distance = np.linalg.norm(np.array([Xc, Yc]) - np.array([Xt, Yt]))
    return -distance  # The closer, the higher the reward

def calculate_circumcenter( own_position, neighbor1_pos, neighbor2_pos):
    """
    Calculate circumcenter given three points
    
    Args:
        own_position (np.array): [x, y] coordinates of the agent
        neighbor1_pos (np.array): [x, y] coordinates of first neighbor
        neighbor2_pos (np.array): [x, y] coordinates of second neighbor
        
    Returns:
        np.array: [x, y] coordinates of circumcenter
    """
    # Convert points to numpy arrays if they aren't already
    p1 = np.array(own_position)
    p2 = np.array(neighbor1_pos)
    p3 = np.array(neighbor2_pos)
    
    # Midpoints of two sides
    mid1 = (p1 + p2) / 2
    mid2 = (p2 + p3) / 2
    
    # Slopes of perpendicular bisectors
    if p2[0] - p1[0] != 0:
        slope1 = -(p2[0] - p1[0]) / (p2[1] - p1[1])
    else:
        slope1 = float('inf')
        
    if p3[0] - p2[0] != 0:
        slope2 = -(p3[0] - p2[0]) / (p3[1] - p2[1])
    else:
        slope2 = float('inf')
        
    # Handle special cases of vertical lines
    if slope1 == float('inf'):
        x = mid1[0]
        y = slope2 * (x - mid2[0]) + mid2[1]
    elif slope2 == float('inf'):
        x = mid2[0]
        y = slope1 * (x - mid1[0]) + mid1[1]
    else:
        # Solve system of equations to find intersection
        x = (mid2[1] - mid1[1] + slope1*mid1[0] - slope2*mid2[0]) / (slope1 - slope2)
        y = slope1 * (x - mid1[0]) + mid1[1]
        
    return np.array([x, y])


# def calculate_reward( circumcenter, original_center, current_pos):
#     """
#     Calculate reward based on how well the circumcenter matches the original center
#     and how well the agent maintains the desired radius
    
#     Args:
#         circumcenter (np.array): Calculated circumcenter
#         original_center (np.array): Target center of formation
#         current_pos (np.array): Current position of the agent
        
#     Returns:
#         float: Reward value
#     """
#     # Distance between calculated circumcenter and original center
#     center_error = np.linalg.norm(circumcenter - original_center)
    
#     # Distance between agent and calculated circumcenter
#     current_radius = np.linalg.norm(current_pos - circumcenter)
#     radius_error = abs(current_radius - target_radius)
    
#     # Combine both errors with weights
#     w1, w2 = 0.6, 0.4  # Weights can be adjusted
#     total_error = w1 * center_error + w2 * radius_error
    
#     # Convert error to reward (negative exponential to keep reward positive)
#     reward = np.exp(-total_error)
    
#     return reward

# Example Usage
robot_position = (2, 3)
neighbor_1 = (3, 5)
neighbor_2 = (1, 4)

target_center = (1, 0.5)  # The actual center of the desired circle

circumcenter_chat = compute_circumcenter(robot_position, neighbor_1, neighbor_2)

# Example positions
own_pos = np.array([2.0, 3.0])
neighbor1 = np.array([3.0, 5.0])
neighbor2 = np.array([1.0, 4.0])
original_center = np.array([1.0, 0.5])

# Calculate circumcenter
circumcenter_claude = calculate_circumcenter(own_pos, neighbor1, neighbor2)

# reward = reward_function(circumcenter, target_center)

print("Computed Circumcenter:", circumcenter_chat)
# print("Reward:", reward)
print("Claude's circumcenter:", circumcenter_claude)