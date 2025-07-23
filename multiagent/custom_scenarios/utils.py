from typing import Optional, Tuple, List
import math
import numpy as np

from multiagent.core import EntityDynamicsType, World, Agent, Landmark, Entity, Wall, BaseEntityState

## create a line of goal positions
# def create_line_of_goals(num_goals: int, goal_spacing: float, goal_y: float, goal_radius: float) -> List[Tuple[float, float]]:
#     goals = []
#     for i in range(num_goals):
#         goal_x = -1.0 + (i + 1) * goal_spacing
#         goals.append((goal_x, goal_y))
#     return goals

# ##Linear Landmark Placement
# def set_landmarks_in_line(self, world):
#     # Calculate spacing between landmarks
#     total_width = self.world_size * 0.8  # Use 80% of world width for better margins
#     spacing = total_width / (self.num_landmarks - 1) if self.num_landmarks > 1 else 0
	
#     # Starting x position (leftmost landmark)
#     start_x = -total_width / 2
	
#     # Place landmarks in a line
#     for i in range(self.num_landmarks):
#         # Calculate position along the line
#         x_pos = start_x + (i * spacing)
#         # Set y position to middle of world (0 assuming world center is origin)
#         pos = np.array([x_pos, 0.0]) if world.dim_p == 2 else np.array([x_pos, 0.0, 0.0])
		
#         # Set landmark position
#         world.landmarks[i].state.p_pos = pos
#         world.landmarks[i].state.reset_velocity()
	
#     # Update landmark positions arrays
#     self.landmark_poses = np.array([landmark.state.p_pos for landmark in world.landmarks])
#     self.landmark_poses_occupied = np.zeros(self.num_agents)
#     self.landmark_poses_updated = np.array([landmark.state.p_pos for landmark in world.landmarks])
#     self.agent_id_updated = np.arange(self.num_agents)
# @staticmethod
# @jit(nopython=True)
def get_rotated_position_from_relative(relative_position: np.ndarray,
	reference_heading: float) -> np.ndarray:
	# returns relative position from the reference state.
	assert relative_position.shape == (2,), "relative_position should be a 2D array."
	rot_matrix = np.array([
		[np.cos(reference_heading), np.sin(reference_heading)],
		[-np.sin(reference_heading),  np.cos(reference_heading)]
	])
	relative_position_rotated = np.dot(rot_matrix, relative_position)
	return relative_position_rotated

def set_landmarks_in_line(self, world, line_angle=0, start_pos=None, end_pos=None):
	"""
	Place landmarks in a straight line between start and end positions.
	
	Args:
		world: World object containing landmarks
		line_angle: Angle of the line in radians (0 is horizontal, pi/2 is vertical)
		start_pos: Starting position of the line (if None, will be calculated)
		end_pos: Ending position of the line (if None, will be calculated)
	"""

	# If start and end positions are not provided, calculate them
	if start_pos is None or end_pos is None:
		# Calculate line length based on world size and number of landmarks
		line_length = self.world_size * 0.6  # Use 60% of world size for line length
		
		# Calculate center of the world
		center = np.zeros(world.dim_p)
		
		# Calculate start and end positions based on angle
		half_length = line_length / 2
		start_pos = center + np.array([
			-half_length * np.cos(line_angle),
			-half_length * np.sin(line_angle)
		])
		end_pos = center + np.array([
			half_length * np.cos(line_angle),
			half_length * np.sin(line_angle)
		])

	# Calculate positions along the line
	positions = np.linspace(start_pos, end_pos, self.num_landmarks)
	# Place landmarks
	for i in range(self.num_landmarks):
		pos = positions[i]
		goal_size = world.landmarks[i].size
		
		# Check for obstacle collisions
		if self.is_obstacle_collision(pos, goal_size, world):
			raise ValueError(f"Landmark {i} collides with obstacle at position {pos}")
			
		# Set landmark position
		world.landmarks[i].state.p_pos = pos
		world.landmarks[i].state.reset_velocity()
		# print(i, "world.landmarks[i].state.p_pos", world.landmarks[i].state.p_pos)


	
	# Store line parameters for reference
	self.line_params = {
		'start_pos': start_pos,
		'end_pos': end_pos,
		'angle': line_angle
	}

##Random Landmark Placement
def set_landmarks_random(self, world):
	##set landmarks (goals) at random positions not colliding with obstacles 
	##and also check collisions with already placed goals
	num_goals_added = 0

	while True:
		if num_goals_added == self.num_landmarks:
			break

		# for random pos
		random_pos = 0.7 * np.random.uniform(-self.world_size/2, 
											self.world_size/2, 
											world.dim_p)


		goal_size = world.landmarks[num_goals_added].size
		obs_collision = self.is_obstacle_collision(random_pos, goal_size, world)
		landmark_collision = self.is_landmark_collision(random_pos, 
											goal_size, 
											world.landmarks[:num_goals_added])
		if not landmark_collision and not obs_collision:
		# if not landmark_collision:
			world.landmarks[num_goals_added].state.p_pos = random_pos
			world.landmarks[num_goals_added].state.reset_velocity()
			num_goals_added += 1

	self.landmark_poses = np.array([landmark.state.p_pos for landmark in world.landmarks])
	self.landmark_poses_occupied = np.zeros(self.num_agents)
	self.landmark_poses_updated = np.array([landmark.state.p_pos for landmark in world.landmarks])


##Random Landmark Placement
def set_landmarks_in_point(self, world, tube_angle, tube_endpoints):
	##set landmarks (goals) at random positions not colliding with obstacles 
	##and also check collisions with already placed goals
	num_goals_added = 0
	# print("tube_endpoints", tube_endpoints)
	# print("tube_angle", tube_angle)

	while True:
		if num_goals_added == self.num_landmarks:
			break

		# Set relative position for the landmark (e.g., offset from tube entrance)
		relative_pos = np.array([0.0, -self.world_size/3])
		# print("relative_pos", relative_pos)
		# Rotate by tube_angle
		rotated_pos = get_rotated_position_from_relative(relative_pos, tube_angle)
		# print("rotated_pos", rotated_pos)
		# Translate by tube_endpoints (tube exit position)
		landmark_pos = np.array(tube_endpoints) + rotated_pos

		goal_size = world.landmarks[num_goals_added].size
		# Optionally check for collisions here
		world.landmarks[num_goals_added].state.p_pos = landmark_pos
		world.landmarks[num_goals_added].state.reset_velocity()
		num_goals_added += 1

	self.landmark_poses = np.array([landmark.state.p_pos for landmark in world.landmarks])
	self.landmark_poses_occupied = np.zeros(self.num_agents)
	self.landmark_poses_updated = np.array([landmark.state.p_pos for landmark in world.landmarks])

##Random Landmark Placement
def set_landmarks_in_point_seq(self, world, tube_endpoints, agent_id, tube_choice):
	##set landmarks (goals) at random positions not colliding with obstacles 
	##and also check collisions with already placed goals
	# num_goals_added = 0



		# for random pos
		# random_pos = 0.1 * np.random.uniform(-self.world_size/2, 
		#                                     self.world_size/2, 
		#                                     world.dim_p)
		# random_pos += tube_endpoints
	# print("tube_choice", tube_choice)
	# print("agent_id", agent_id)
	if tube_choice %2== 1:
		random_pos = np.array([tube_endpoints[0]+0.5*self.world_size,tube_endpoints[1]])
	else:
		random_pos = np.array([tube_endpoints[0]- 0.5*self.world_size,tube_endpoints[1]])


	goal_size = world.landmarks[agent_id].size
		# obs_collision = self.is_obstacle_collision(random_pos, goal_size, world)
		# landmark_collision = self.is_landmark_collision(random_pos, 
		#                                     goal_size, 
		#                                     world.landmarks[:num_goals_added])
		# if not landmark_collision and not obs_collision:
		# if not landmark_collision:
	world.landmarks[agent_id].state.p_pos = random_pos
	world.landmarks[agent_id].state.reset_velocity()

	self.landmark_poses = np.array([landmark.state.p_pos for landmark in world.landmarks])
	self.landmark_poses_occupied = np.zeros(self.num_agents)
	self.landmark_poses_updated = np.array([landmark.state.p_pos for landmark in world.landmarks])


def set_landmarks_in_circle(self, world, center=np.array([0.0, 0.0]), radius=1.0):
	"""
	Place landmarks in a circular formation
	Args:
		world: The world object containing landmarks
		center: Center point of the circle (default: origin)
		radius: Radius of the circle (default: 1.0)
	"""
	# Calculate angle between each landmark
	angle_step = 2 * np.pi / self.num_landmarks
	
	# Place landmarks in a circle
	for i in range(self.num_landmarks):
		# Calculate angle for current landmark
		angle = i * angle_step
		
		# Calculate position using parametric equation of circle
		# x = center_x + r * cos(θ)
		# y = center_y + r * sin(θ)
		x_pos = center[0] + radius * np.cos(angle)
		y_pos = center[1] + radius * np.sin(angle)
		
		# Create position array based on world dimensions
		if world.dim_p == 2:
			pos = np.array([x_pos, y_pos])
		else:  # 3D world
			pos = np.array([x_pos, y_pos, 0.0])  # Placing circle in x-y plane
		
		# Set landmark position
		world.landmarks[i].state.p_pos = pos
		world.landmarks[i].state.reset_velocity()
	
	# Update landmark positions arrays
	self.landmark_poses = np.array([landmark.state.p_pos for landmark in world.landmarks])
	self.landmark_poses_occupied = np.zeros(self.num_agents)
	self.landmark_poses_updated = np.array([landmark.state.p_pos for landmark in world.landmarks])
	self.agent_id_updated = np.arange(self.num_agents)

def get_relative_position_from_reference(query_position: np.ndarray,
	reference_position: np.ndarray, reference_heading: float):
	# returns relative position from the reference state.
	assert query_position.shape == (2,), "query_position should be a 2D array."
	assert reference_position.shape == (2,), "reference_position should be a 2D array."
	relative_position = query_position - reference_position
	rot_matrix = np.array([[np.cos(reference_heading), np.sin(reference_heading)], [-np.sin(reference_heading), np.cos(reference_heading)]])
	relative_position_rotated = np.dot(rot_matrix, relative_position)
	return relative_position_rotated



def get_agent_observation_relative_with_heading(agent_position: np.ndarray, agent_heading: float, agent_speed: float,
												goal_position: np.ndarray, goal_heading: float, goal_speed: float):
	# Returns observations relatively defined with respect to the agent's state.
	# Used for kinematic vehicle type where heading is important (non-holonomic).
	assert goal_heading is not None, "goal_heading should not be None."
	assert goal_speed is not None, "goal_speed should not be None."

	relative_goal_position = get_relative_position_from_reference(goal_position, agent_position, agent_heading)
	relative_goal_heading = goal_heading - agent_heading
	relative_goal_heading_sincos = np.array([np.sin(relative_goal_heading), np.cos(relative_goal_heading)])
	obs = np.concatenate([np.array([agent_speed]), relative_goal_position, relative_goal_heading_sincos, np.array([goal_speed])])
	print("relative_goal_position: ", relative_goal_position, "relative_goal_heading: ", relative_goal_heading, "relative_goal_heading_sincos: ", relative_goal_heading_sincos)
	print("obs: ", obs)
	return obs