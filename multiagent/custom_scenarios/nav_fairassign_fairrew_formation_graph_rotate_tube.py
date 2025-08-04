"""
	Navigation for `n` agents to `n` goals from random initial positions
	With random obstacles added in the environment
	Each agent is destined to get to its own goal unlike
	`simple_spread.py` where any agent can get to any goal (check `reward()`)
"""
from typing import Optional, Tuple, List
import argparse
import numpy as np
from numpy import ndarray as arr
from scipy import sparse

import scipy.spatial.distance as dist
import os,sys
sys.path.append(os.path.abspath(os.getcwd()))

from multiagent.core import EntityDynamicsType, World, Agent, Landmark, Entity, Wall
from multiagent.scenario import BaseScenario
from multiagent.config import UnicycleVehicleConfig, DoubleIntegratorConfig
from scipy.optimize import linear_sum_assignment
from multiagent.custom_scenarios.utils import *
from marl_fair_assign import solve_fair_assignment

entity_mapping = {'agent': 0, 'landmark': 1, 'obstacle':2, 'wall':3}


def get_thetas(poses):
	# compute angle (0,2pi) from horizontal
	thetas = [None] * len(poses)
	for i in range(len(poses)):
		# (y,x)
		thetas[i] = find_angle(poses[i])
	return thetas


def find_angle(pose):
	# compute angle from horizontal
	angle = np.arctan2(pose[1], pose[0])
	if angle < 0:
		angle += 2 * np.pi
	return angle


def leaky_ReLU(x):
  data = np.max(max(0.01*x,x))
  return np.array(data, dtype=float)


class Scenario(BaseScenario):
	def make_world(self, args:argparse.Namespace) -> World:
		"""
			Parameters in args
			––––––––––––––––––
			• num_agents: int
				Number of agents in the environment
				NOTE: this is equal to the number of goal positions
			• num_obstacles: int
				Number of num_obstacles obstacles
			• collaborative: bool
				If True then reward for all agents is sum(reward_i)
				If False then reward for each agent is what it gets individually
			• max_speed: Optional[float]
				Maximum speed for agents
				NOTE: Even if this is None, the max speed achieved in discrete 
				action space is 2, so might as well put it as 2 in experiments
				TODO: make list for this and add this in the state
			• collision_rew: float
				The reward to be negated for collisions with other agents and 
				obstacles
			• goal_rew: float
				The reward to be added if agent reaches the goal
			• min_dist_thresh: float
				The minimum distance threshold to classify whether agent has 
				reached the goal or not
			• use_dones: bool
				Whether we want to use the 'done=True' when agent has reached 
				the goal or just return False like the `simple.py` or 
				`simple_spread.py`
			• episode_length: int
				Episode length after which environment is technically reset()
				This determines when `done=True` for done_callback
			• graph_feat_type: str
				The method in which the node/edge features are encoded
				Choices: ['global', 'relative']
					If 'global': 
						• node features are global [pos, vel, goal, entity-type]
						• edge features are relative distances (just magnitude)
						• 
					If 'relative':
						• TODO decide how to encode stuff

			• max_edge_dist: float
				Maximum distance to consider to connect the nodes in the graph
		"""
		# pull params from args
		if not hasattr(self, 'world_size'):
			self.world_size = args.world_size
		self.num_agents = args.num_agents
		self.num_scripted_agents = args.num_scripted_agents
		self.num_obstacles = args.num_obstacles
		self.collaborative = args.collaborative
		self.max_speed = args.max_speed
		self.collision_rew = args.collision_rew
		self.goal_rew = args.goal_rew
		self.min_dist_thresh = args.min_dist_thresh
		self.min_obs_dist = args.min_obs_dist

		self.use_dones = args.use_dones
		self.episode_length = args.episode_length
		self.target_radius = 0.5  # fixing the target radius for now
		self.ideal_theta_separation = (
			2 * np.pi
		) / self.num_agents  # ideal theta difference between two agents

		# fairness args
		self.fair_wt = args.fair_wt
		self.fair_rew = args.fair_rew

		self.formation_type = args.formation_type

		# create heatmap matrix to determine the goal agent pairs
		self.goal_reached = -1*np.ones(self.num_agents)
		self.wrong_goal_reached = np.zeros(self.num_agents)
		self.goal_matched = np.zeros(self.num_agents)
		if args.dynamics_type == 'unicycle_vehicle':
			self.dynamics_type = EntityDynamicsType.UnicycleVehicleXY
			self.config_class = UnicycleVehicleConfig
			self.min_turn_radius = 0.5 * (UnicycleVehicleConfig.V_MAX + UnicycleVehicleConfig.V_MIN) / UnicycleVehicleConfig.ANGULAR_RATE_MAX

		elif args.dynamics_type == 'double_integrator':
			self.dynamics_type = EntityDynamicsType.DoubleIntegratorXY
			self.config_class = DoubleIntegratorConfig
			self.min_turn_radius = 0.0
		else:
			raise NotImplementedError
		self.coordination_range = self.config_class.COMMUNICATION_RANGE
		self.min_dist_thresh = self.config_class.DISTANCE_TO_GOAL_THRESHOLD
		self.separation_distance = self.config_class.COLLISION_DISTANCE

		self.phase_reached = np.zeros(self.num_agents)
		# scripted agent dynamics are fixed to double integrator for now (as it was originally)
		scripted_agent_dynamics_type = EntityDynamicsType.DoubleIntegratorXY
		# if not hasattr(args, 'max_edge_dist'):
		# 	self.max_edge_dist = 1
		# 	print('_'*60)
		# 	print(f"Max Edge Distance for graphs not specified. "
		# 			f"Setting it to {self.max_edge_dist}")
		# 	print('_'*60)
		# else:
		# 	self.max_edge_dist = args.max_edge_dist
		self.max_edge_dist = self.coordination_range
		####################
		world = World(dynamics_type=self.dynamics_type)
		# graph related attributes
		world.cache_dists = True # cache distance between all entities
		world.graph_mode = True
		world.graph_feat_type = args.graph_feat_type
		world.world_length = args.episode_length
		# metrics to keep track of
		world.current_time_step = 0
		# to track time required to reach goal
		world.times_required = -1 * np.ones(self.num_agents)
		world.dists_to_goal = -1 * np.ones(self.num_agents)
		# set any world properties
		world.dim_c = 2
		self.num_landmarks = args.num_landmarks # no. of goals need not equal to no. of agents
		num_scripted_agents_goals = self.num_scripted_agents
		world.collaborative = args.collaborative
		#############
		## determine the number of actions from arguments
		world.total_actions = args.total_actions
		#############
		# add agents
		global_id = 0
		world.agents = [Agent(self.dynamics_type) for i in range(self.num_agents)]
		world.scripted_agents = [Agent(scripted_agent_dynamics_type) for _ in range(self.num_scripted_agents)]
		for i, agent in enumerate(world.agents + world.scripted_agents):
			agent.id = i
			agent.name = f'agent {i}'
			agent.collide = True
			agent.silent = True
			agent.global_id = global_id
			global_id += 1
			# NOTE not changing size of agent because of some edge cases; 
			# TODO have to change this later
			# agent.size = 0.15
			agent.max_speed = self.max_speed
		# add landmarks (goals)
		world.landmarks = [Landmark() for i in range(self.num_landmarks)]
		world.scripted_agents_goals = [Landmark() for i in range(num_scripted_agents_goals)]
		for i, landmark in enumerate(world.landmarks):
			landmark.id = i
			landmark.name = f'landmark {i}'
			landmark.collide = False
			landmark.movable = False
			landmark.global_id = global_id
			global_id += 1
		# add obstacles
		world.obstacles = [Landmark() for i in range(self.num_obstacles)]
		for i, obstacle in enumerate(world.obstacles):
			obstacle.name = f'obstacle {i}'
			obstacle.collide = True
			obstacle.movable = False
			obstacle.global_id = global_id
			global_id += 1
		## add wall
		# num obstacles per wall is twice the length of the wall
		wall_length = np.random.uniform(0.2, 0.4)
		self.wall_length = wall_length * self.world_size/4

		self.num_walls = args.num_walls
		world.walls = [Wall() for i in range(self.num_walls)]
		for i, wall in enumerate(world.walls):
			wall.id = i
			wall.name = f'wall {i}'
			wall.collide = True
			wall.movable = False
			wall.global_id = global_id
			global_id += 1

			
		self.zeroshift = args.zeroshift
		self.reset_world(world)
		world.world_size = self.world_size
		# world.world_aspect_ratio = self.world_aspect_ratio
		return world

	def reset_world(self, world:World) -> None:

		# metrics to keep track of
		world.current_time_step = 0
		world.simulation_time = 0.0
		# to track time required to reach goal
		world.times_required = -1 * np.ones(self.num_agents)
		world.dists_to_goal = -1 * np.ones(self.num_agents)

		# track distance left to the goal
		world.dist_left_to_goal = -1 * np.ones(self.num_agents)
		# number of times agents collide with stuff
		world.num_obstacle_collisions = np.zeros(self.num_agents)
		world.num_goal_collisions = np.zeros(self.num_agents)
		world.num_agent_collisions = np.zeros(self.num_agents)
		world.agent_dist_traveled = np.zeros(self.num_agents)

		self.goal_match_index = np.arange(self.num_agents)
		self.goal_history = -1*np.ones((self.num_agents))
		self.goal_reached = -1*np.ones(self.num_agents)

		self.agent_dist_traveled = np.zeros(self.num_agents)
		self.agent_time_taken = np.zeros(self.num_agents)
		wall_length = np.random.uniform(0.2, 0.8)
		self.wall_length = wall_length * self.world_size/4


		self.phase_reached = np.zeros(self.num_agents)


		#################### set colours ####################
		# set colours for agents
		for i, agent in enumerate(world.agents):
			agent.color = np.array([0.85, 0.35, 0.35])
			if i%4 == 0:
				agent.color = np.array([0.85, 0.35, 0.35])
			elif i%4 == 1:
				agent.color = np.array([0.35, 0.85, 0.35])
			elif i%4 == 2:
				agent.color = np.array([0.35, 0.35, 0.85])
			else:
				agent.color = np.array([0.85, 0.85, 0.25])
			# if i == 0:
			# 	agent.color = np.array([0.15, 0.75, 0.65])
			agent.state.p_dist = 0.0
			agent.state.time = 0.0
		# set colours for scripted agents
		for i, agent in enumerate(world.scripted_agents):
			agent.color = np.array([0.15, 0.15, 0.15])
		# set colours for landmarks
		for i, landmark in enumerate(world.landmarks):
			if i%4 == 0:
				landmark.color = np.array([0.85, 0.35, 0.35])
			elif i%4 == 1:
				landmark.color = np.array([0.35, 0.85, 0.35])
			elif i%4 == 2:
				landmark.color =  np.array([0.35, 0.35, 0.85])
			else:
				landmark.color = np.array([0.85, 0.85, 0.25])
			# if i == 0:
			# 	landmark.color = np.array([0.15, 0.75, 0.65])
		# set colours for scripted agents goals
		for i, landmark in enumerate(world.scripted_agents_goals):
			landmark.color = np.array([0.15, 0.95, 0.15])
		# set colours for obstacles
		for i, obstacle in enumerate(world.obstacles):
			obstacle.color = np.array([0.25, 0.25, 0.25])
		# set colours for wall obstacles
		# for i, wall_obstacle in enumerate(world.wall_obstacles):
		# 	wall_obstacle.color = np.array([0.25, 0.25, 0.25])
		#####################################################
		self.random_scenario(world)
		self.initialize_min_time_distance_graph(world)

	def random_scenario(self, world):
		"""
			Randomly place agents and landmarks
		"""
		# Initialize tube parameters
		self.setup_tube_params(world)
		# Calculate cluster center for initial agent positions
		# Position it above tube entrance, perpendicular to tube direction
		tube_angle = world.tube_params['angle']
		entrance = world.tube_params['entrance']
		
		# Calculate cluster region above tube entrance
		cluster_center = entrance + np.array([
			0,  # Same x as entrance
			self.world_size * 0.2  # Offset above entrance
		])
		
		# Set random positions within cluster
		cluster_radius = self.world_size * 0.15  # Size of initial cluster
		num_agents_added = 0
		agents_added = []
		max_attempts = 1000  # Prevent infinite loops
		attempt_count = 0
		
		while num_agents_added < self.num_agents and attempt_count < max_attempts:
			# Generate random position within cluster
			random_offset = np.random.uniform(-cluster_radius, cluster_radius, world.dim_p)
			random_pos = cluster_center + random_offset
			
			# Check if position is within world bounds
			if (abs(random_pos[0]) > self.world_size/2 * 0.9 or 
				abs(random_pos[1]) > self.world_size/2 * 0.9):
				attempt_count += 1
				continue
			
			agent_size = world.agents[num_agents_added].size
			obs_collision = self.is_obstacle_collision(random_pos, agent_size, world)
			agent_collision = self.check_agent_collision(random_pos, agent_size, agents_added)
			
			if not obs_collision and not agent_collision:
				world.agents[num_agents_added].state.p_pos = random_pos
				world.agents[num_agents_added].state.reset_velocity()
				world.agents[num_agents_added].state.c = np.zeros(world.dim_c)
				world.agents[num_agents_added].status = False
				agents_added.append(world.agents[num_agents_added])
				num_agents_added += 1
			
			attempt_count += 1
		
		self.agent_id_updated = np.arange(self.num_agents)
		
		if self.formation_type == 'line':
			# Calculate goal line parameters perpendicular to tube at bottom of world
			goal_line_length = self.world_size * 0.8  # 40% of world size for goal line
			half_length = goal_line_length / 2
			
			# Calculate goal line angle perpendicular to tube
			goal_line_angle = tube_angle
			
			# Calculate center point for goal line at bottom of world
			goal_center_y = -self.world_size/2 * 0.9  # Place near bottom of world
			
			# Project tube angle to find goal center x
			# This ensures the goal line center aligns with tube trajectory
			tube_direction = np.array([np.sin(tube_angle), -np.cos(tube_angle)])  # Direction vector of tube
			projection_distance = (world.tube_params['exit'][1] - goal_center_y) / tube_direction[1]
			goal_center_x = world.tube_params['exit'][0] + tube_direction[0] * projection_distance
			
			goal_center = np.array([goal_center_x, goal_center_y])
			# Calculate start and end positions for the goal line
			goal_line_start = goal_center + np.array([
				-half_length * np.cos(goal_line_angle),
				-half_length * np.sin(goal_line_angle)
			])
			
			goal_line_end = goal_center + np.array([
				half_length * np.cos(goal_line_angle),
				half_length * np.sin(goal_line_angle)
			])
			
			# Set landmarks in line formation
			set_landmarks_in_line(self,
				world,
				line_angle=goal_line_angle,
				start_pos=goal_line_start,
				end_pos=goal_line_end
			)
		# elif self.formation_type == 'circle':
		# 	set_landmarks_in_circle(self, world, center=np.array([0.0, 0.0]), radius=self.world_size/3)
		# elif self.formation_type == 'random':
		# 	set_landmarks_random(self, world)
		# else:
		# 	raise NotImplementedError

		# Update landmark poses arrays
		self.landmark_poses = np.array([landmark.state.p_pos for landmark in world.landmarks])
		self.landmark_poses_occupied = np.zeros(self.num_agents)
		self.landmark_poses_updated = np.array([landmark.state.p_pos for landmark in world.landmarks])
		self.agent_id_updated = np.arange(self.num_agents)
		#####################################################

		############ find minimum times to goals ############
		if self.max_speed is not None:
			for agent in world.agents:
				self.min_time(agent, world)
		#####################################################
		# ############ update the cached distances ############
		# world.calculate_distances()
		# self.update_graph(world)
		# ####################################################
		
		# ########### set fair goals for each agent ###########
		# costs = dist.cdist(agent_pos, self.landmark_poses)
		# # print("costs",costs)
		# x, objs = solve_fair_assignment(costs)
		# # print('x',x,"objs", objs)
		# self.goal_match_index = np.where(x==1)[1]
		# if self.goal_match_index.size == 0 or self.goal_match_index.shape != (self.num_agents,):
		# 	self.goal_match_index = np.arange(self.num_agents)

		# #####################################################

		############ check fairness metric match ###########


	def setup_tube_params(self, world):
		"""
		Set up tube parameters with random rotation angle
		"""
		# Calculate tube width based on number of agents
		self.tube_width = max(
			self.num_agents * world.agents[0].size * 2.5,  # Width based on agents
			self.world_size * 0.15  # Minimum width
		)
		
		# Generate random angle between -45 and 45 degrees (π/4 radians)
		random_angle = np.random.uniform(-np.pi/4, np.pi/4)
		
		# Calculate tube length
		tube_length = self.world_size * 0.8  # Use 80% of world size for tube length
		
		# Calculate center point of the world
		world_center = np.array([0, 0])
		
		# Calculate entrance and exit points using rotation
		# Start with vertical positions
		base_entrance = np.array([0, tube_length/4])  # Start above center
		base_exit = np.array([0, -tube_length/4])     # End below center
		
		# Create rotation matrix
		rotation_matrix = np.array([
			[np.cos(random_angle), -np.sin(random_angle)],
			[np.sin(random_angle), np.cos(random_angle)]
		])
		
		# Apply rotation to entrance and exit points
		entrance = world_center + rotation_matrix @ base_entrance
		exit = world_center + rotation_matrix @ base_exit
		
		# Store tube parameters
		world.tube_params = {
			'entrance': entrance,
			'exit': exit,
			'width': self.tube_width,
			'angle': random_angle,
			'length': tube_length
		}
		
		# Calculate perpendicular direction for formation lines
		perpendicular_angle = random_angle + np.pi/2
		formation_direction = np.array([
			np.cos(perpendicular_angle),
			np.sin(perpendicular_angle)
		])
		
		# Calculate line formation parameters
		line_length = self.tube_width * 0.8  # Slightly smaller than tube width
		half_line = (line_length/2) * formation_direction
		
		# Pre-tube formation line (above entrance)
		pre_tube_center = entrance + rotation_matrix @ np.array([0, self.world_size * 0.15])
		self.pre_tube_line = {
			'start_pos': pre_tube_center - half_line,
			'end_pos': pre_tube_center + half_line,
			'angle': perpendicular_angle
		}
		
		# Post-tube target line (below exit)
		post_tube_center = exit - rotation_matrix @ np.array([0, self.world_size * 0.15])
		self.post_tube_line = {
			'start_pos': post_tube_center - half_line,
			'end_pos': post_tube_center + half_line,
			'angle': perpendicular_angle
		}
		
		# Store additional parameters
		world.tube_params.update({
			'pre_tube_line': self.pre_tube_line,
			'post_tube_line': self.post_tube_line,
			'rotation_matrix': rotation_matrix,  # Store for potential future use
			'formation_direction': formation_direction  # Direction for agent lineup
		})
	
	def is_in_tube(self, world: World, pos):
		"""
		Updated helper function to check if position is inside rotated tube
		"""
		entrance = world.tube_params['entrance']
		exit = world.tube_params['exit']
		width = world.tube_params['width']
		angle = world.tube_params['angle']
		
		# Calculate tube direction vector
		tube_direction = exit - entrance
		tube_length = np.linalg.norm(tube_direction)
		tube_direction = tube_direction / tube_length
		
		# Calculate vector from entrance to position
		pos_vector = pos - entrance
		
		# Project position onto tube direction
		projection_length = np.dot(pos_vector, tube_direction)
		
		# Check if point is between entrance and exit
		if projection_length < 0 or projection_length > tube_length:
			return False
			
		# Calculate perpendicular distance to tube centerline
		perpendicular_vector = pos_vector - projection_length * tube_direction
		perpendicular_distance = np.linalg.norm(perpendicular_vector)
		
		# Check if point is within tube width
		return perpendicular_distance <= width/2
	
	def get_agent_phase(self, agent: Agent, world: World):
		"""
		Updated helper function to determine agent's current phase with rotated tube
		"""
		pos = agent.state.p_pos
		in_tube = self.is_in_tube(world, pos)
		
		# Calculate vector from exit to position
		exit_to_pos = pos - world.tube_params['exit']
		tube_direction = world.tube_params['exit'] - world.tube_params['entrance']
		tube_direction = tube_direction / np.linalg.norm(tube_direction)
		
		# Project onto tube direction to see if we've passed the tube
		passed_tube = np.dot(exit_to_pos, tube_direction) < 0
		
		if not in_tube and not passed_tube:
			return 0  # Pre-tube phase
		elif in_tube:
			return 1  # In-tube phase
		else:
			return 2  # Post-tube phase

	def initialize_min_time_distance_graph(self, world):
		for agent in world.agents:
			self.min_time(agent, world)
		world.calculate_distances()
		self.update_graph(world)

	def info_callback(self, agent:Agent, world:World) -> Tuple:
		# TODO modify this 

		world.dists = np.array([np.linalg.norm(agent.state.p_pos - l.state.p_pos) for l in world.landmarks])
		
		nearest_landmark = np.argmin(world.dists)
		dist_to_goal = world.dists[nearest_landmark]
		if dist_to_goal < self.min_dist_thresh and (nearest_landmark != self.goal_reached[agent.id] and self.goal_reached[agent.id] != -1):
			# print("AGENT", agent.id, "reached NEW goal",world.dist_left_to_goal[agent.id])
			self.goal_reached[agent.id] = nearest_landmark
			world.dist_left_to_goal[agent.id] = dist_to_goal
		# only update times_required for the first time it reaches the goal
		if dist_to_goal < self.min_dist_thresh and (world.times_required[agent.id] == -1):
			# print("agent", agent.id, "reached goal",world.dist_left_to_goal[agent.id])
			world.times_required[agent.id] = world.current_time_step * world.dt
			world.dists_to_goal[agent.id] = agent.state.p_dist
			world.dist_left_to_goal[agent.id] = dist_to_goal
			self.goal_reached[agent.id] = nearest_landmark
			# print("dist to goal",world.dists_to_goal[agent.id],world.dists_to_goal)
		if world.times_required[agent.id] == -1:
			# print("agent", agent.id, "not reached goal yet",world.dist_left_to_goal[agent.id])
			world.dists_to_goal[agent.id] = agent.state.p_dist
			world.dist_left_to_goal[agent.id] = dist_to_goal
		if dist_to_goal > self.min_dist_thresh and world.times_required[agent.id] != -1:
			# print("AGENT", agent.id, "left goal",dist_to_goal,world.dist_left_to_goal[agent.id])
			world.dists_to_goal[agent.id] = agent.state.p_dist
			world.times_required[agent.id] = world.current_time_step * world.dt
			world.dist_left_to_goal[agent.id] = dist_to_goal
		if dist_to_goal < self.min_dist_thresh and (nearest_landmark == self.goal_reached[agent.id]):
			# print("AGENT", agent.id, "on SAME goal",world.dist_left_to_goal[agent.id])
			## TODO: How to update time as well?
			world.dist_left_to_goal[agent.id] = dist_to_goal
			self.goal_reached[agent.id] = nearest_landmark

			# print("dist is",world.dists_to_goal[agent.id])

		if agent.collide:
			if self.is_obstacle_collision(agent.state.p_pos, agent.size, world):
				world.num_obstacle_collisions[agent.id] += 1
			for a in world.agents:
				if a is agent:
					self.agent_dist_traveled[a.id] = a.state.p_dist
					self.agent_time_taken[a.id] = a.state.time
					continue
				if self.is_collision(agent, a):
					world.num_agent_collisions[agent.id] += 1


		world.dist_traveled_mean = np.mean(world.dists_to_goal)
		world.dist_traveled_stddev = np.std(world.dists_to_goal)
		# create the mean and stddev for time taken
		world.time_taken_mean = np.mean(world.times_required)
		world.time_taken_stddev = np.std(world.times_required)

		agent_info = {
			'Dist_to_goal': world.dist_left_to_goal[agent.id],
			'Time_req_to_goal': world.times_required[agent.id],
			# NOTE: total agent collisions is half since we are double counting. # EDIT corrected this.
			'Num_agent_collisions': world.num_agent_collisions[agent.id], 
			'Num_obst_collisions': world.num_obstacle_collisions[agent.id],
			'Distance_mean': world.dist_traveled_mean, 
			'Distance_variance': world.dist_traveled_stddev,
			'Mean_by_variance': world.dist_traveled_mean/(world.dist_traveled_stddev+0.0001),
			'Dists_traveled': world.dists_to_goal[agent.id],
			'Time_taken': world.times_required[agent.id],
			# Time mean and stddev
			'Time_mean': world.time_taken_mean,
			'Time_stddev': world.time_taken_stddev,
			'Time_mean_by_stddev': world.time_taken_mean/(world.time_taken_stddev+0.0001),


		}
		if self.max_speed is not None:
			agent_info['Min_time_to_goal'] = agent.goal_min_time
		return agent_info

	# check collision of entity with obstacles and walls
	def is_obstacle_collision(self, pos, entity_size:float, world:World) -> bool:
		# pos is entity position "entity.state.p_pos"
		collision = False
		for obstacle in world.obstacles:
			delta_pos = obstacle.state.p_pos - pos
			dist = np.linalg.norm(delta_pos)
			dist_min = 2.0*(obstacle.size + entity_size)

			if dist < dist_min:
				collision = True
				break	
		
		# check collision with walls
		for wall in world.walls:
			if wall.orient == 'H':
				# Horizontal wall, check for collision along the y-axis
				if (wall.axis_pos - 1.5*entity_size ) <= pos[1] <= (wall.axis_pos + 1.5*entity_size ):
					if (wall.endpoints[0] - 1.5*entity_size ) <= pos[0] <= (wall.endpoints[1] + 1.5*entity_size ):
						collision = True
						break
			elif wall.orient == 'V':
				# Vertical wall, check for collision along the x-axis
				if (wall.axis_pos - 1.5*entity_size ) <= pos[0] <= (wall.axis_pos + 1.5*entity_size ):
					if (wall.endpoints[0] - 1.5*entity_size ) <= pos[1] <= (wall.endpoints[1] + 1.5*entity_size ):
						collision = True
						break
		return collision
	# check collision of entity with obstacles and walls


	# check collision of agent with other agents
	def check_agent_collision(self, pos, agent_size, agent_added) -> bool:
		collision = False
		if len(agent_added):
			for agent in agent_added:
				delta_pos = agent.state.p_pos - pos
				dist = np.linalg.norm(delta_pos)
				if dist < 1.05*(agent.size + agent_size):
					collision = True
					break
		return collision

	# check collision of agent with another agent
	def is_collision(self, agent1:Agent, agent2:Agent) -> bool:
		delta_pos = agent1.state.p_pos - agent2.state.p_pos
		dist = np.linalg.norm(delta_pos)
		dist_min = 1.05*(agent1.size + agent2.size)
		return True if dist < dist_min else False

	def is_landmark_collision(self, pos, size:float, landmark_list:List) -> bool:
		collision = False
		for landmark in landmark_list:
			delta_pos = landmark.state.p_pos - pos
			dist = np.sqrt(np.sum(np.square(delta_pos)))
			dist_min = 1.2*(size + landmark.size)
			if dist < dist_min:
				collision = True
				break
		return collision

	# get min time required to reach to goal without obstacles
	def min_time(self, agent:Agent, world:World) -> float:
		assert agent.max_speed is not None, "Agent needs to have a max_speed."
		assert agent.max_speed > 0, "Agent max_speed should be positive."
		agent_id = agent.id
		# get the goal associated to this agent
		landmark = world.get_entity(entity_type='landmark', id=self.goal_match_index[agent.id])
		dist = np.sqrt(np.sum(np.square(agent.state.p_pos - 
										landmark.state.p_pos)))
		min_time = dist / agent.max_speed
		agent.goal_min_time = min_time
		return min_time

	# done condition for each agent
	def done(self, agent:Agent, world:World) -> bool:
		# if we are using dones then return appropriate done
		if self.use_dones:
			if world.current_time_step >= world.world_length:
				return True
			else:
				landmark = world.get_entity('landmark',self.goal_match_index[agent.id])
				dist = np.sqrt(np.sum(np.square(agent.state.p_pos - 
												landmark.state.p_pos)))
				if dist < self.min_dist_thresh:
					return True
				else:
					return False
		# it not using dones then return done 
		# only when episode_length is reached
		else:
			if world.current_time_step >= world.world_length:
				return True
			else:
				return False


	def _bipartite_min_dists(self, dists):
		ri, ci = linear_sum_assignment(dists)
		min_dists = dists[ri, ci]
		return min_dists
	


	def reward(self, agent: Agent, world: World) -> float:
		rew = 0
		current_phase = self.get_agent_phase(agent, world)

		# Update the global phase tracker if any agent progresses
		if current_phase > self.phase_reached[agent.id]:
			self.phase_reached[agent.id] = current_phase  # Update max phase reached globally

		if world.dists_to_goal[agent.id] == -1:
			mean_dist,  std_dev_dist, _ = self.collect_dist(world)
			fairness_param = mean_dist/(std_dev_dist+0.0001)
		else:
			fairness_param = world.dist_traveled_mean/(world.dist_traveled_stddev+0.0001)
		if agent.id == 0:

			##### fair assignment to check if it makes it more fair
			agent_pos = [agent.state.p_pos for agent in world.agents]
			costs = dist.cdist(agent_pos, self.landmark_poses)


			x, objs = solve_fair_assignment(costs)
			self.goal_match_index = np.where(x==1)[1]
			if self.goal_match_index.size == 0 or self.goal_match_index.shape != (self.num_agents,):
				self.goal_match_index = np.arange(self.num_agents)

		# Common rewards across all phases
		# Collision penalties
		if agent.collide:
			for a in world.agents:
				if a.id == agent.id:
					continue
				if self.is_collision(a, agent):
					rew -= self.collision_rew
			
			if self.is_obstacle_collision(pos=agent.state.p_pos,
										entity_size=agent.size, 
										world=world):
				rew -= self.collision_rew

		# Calculate tube length
		tube_length = world.tube_params['entrance'][1] - world.tube_params['exit'][1]
		
		# Find agents in front and behind
		agents_sorted_by_y = sorted(world.agents, key=lambda a: a.state.p_pos[1], reverse=True)
		agent_idx = agents_sorted_by_y.index(agent)

		# Get closest agent in front and behind
		front_agent = agents_sorted_by_y[agent_idx - 1] if agent_idx > 0 else None
		back_agent = agents_sorted_by_y[agent_idx + 1] if agent_idx < len(agents_sorted_by_y) - 1 else None
		
		# Calculate desired spacing based on tube length and number of agents
		desired_spacing = tube_length / (len(world.agents) + 1)

		# desired_spacing = world.tube_params['width'] / (len(world.agents) + 1)
		# Reward for line formation
		# neighbor_dists = [np.linalg.norm(other.state.p_pos - agent.state.p_pos) 
		# 				for other in world.agents if other is not agent]
		
		# Track agent's previous phase if not already stored
		if not hasattr(agent, 'previous_phase'):
			agent.previous_phase = 0
		
		# ##Modified phase transition handling
		if current_phase == 2 and current_phase > agent.previous_phase + 1:
			# Only penalize clear phase skips (e.g., 0 to 2)
			rew -= self.goal_rew  # Reduced penalty
			# print(f"Agent {agent.id} penalized for skipping from phase {agent.previous_phase} to {current_phase}")
		if current_phase == agent.previous_phase+1 and self.phase_reached[agent.id] == current_phase:
			# Reward proper phase progression
			rew += self.goal_rew * 0.1  # Positive reward for proper transition
			# print(f"Agent {agent.id} properly progressed from phase {agent.previous_phase} to {current_phase}")
		
		# Store current phase for next step
		agent.previous_phase = current_phase
		# Phase-specific rewards
		# print("Agent",agent.id,"Phase",current_phase)
		if current_phase == 0:  # Pre-tube phase
			# Reward for getting closer to tube entrance
			dist_to_entrance = np.linalg.norm(world.tube_params['entrance'] - agent.state.p_pos)
			rew -= dist_to_entrance
			# print("dist_to_entrance",dist_to_entrance)

			# # Formation reward considering front and back agents
			# spacing_error = 0
			# if front_agent:
			# 	spacing_error += np.abs(np.linalg.norm(front_agent.state.p_pos - agent.state.p_pos) - desired_spacing)
			# if back_agent:
			# 	spacing_error += np.abs(np.linalg.norm(back_agent.state.p_pos - agent.state.p_pos) - desired_spacing)
			# rew -= spacing_error
			# print("Phase 0 spacing_error",spacing_error)
				
		elif current_phase == 1:  # In-tube phase
			# Stronger formation rewards inside tube
			spacing_error = 0
			if front_agent:
				spacing_error += np.abs(np.linalg.norm(front_agent.state.p_pos - agent.state.p_pos) - desired_spacing)
			if back_agent:
				spacing_error += np.abs(np.linalg.norm(back_agent.state.p_pos - agent.state.p_pos) - desired_spacing)
			rew -= spacing_error  # Higher weight for maintaining formation in tube
			# print("Phase 1 spacing_error",spacing_error)
			# Progress through tube
			dist_to_exit = np.linalg.norm(world.tube_params['exit'] - agent.state.p_pos)
			rew -= dist_to_exit
			# print("dist_to_exit",dist_to_exit)
				
		else:  # Post-tube phase
			dist_to_fair_goal = np.linalg.norm(agent.state.p_pos - self.landmark_poses[self.goal_match_index[agent.id]])
			if dist_to_fair_goal < self.min_dist_thresh:
				# print("Agent",agent.id,"reached fair goal")
				if agent.status ==False:
					agent.status = True
					agent.state.reset_velocity()
					rew += self.goal_rew
					# print("Phase 2 Agent",agent.id,"reached fair goal")

			else:
					# print("dist_to_fair_goal",dist_to_fair_goal)
					rew -= dist_to_fair_goal
		
		# # Global formation quality (calculated once per step)
		# if agent.id == 0:
		# 	all_spacings = []
		# 	for a in world.agents:
		# 		a_neighbor_dists = [np.linalg.norm(other.state.p_pos - a.state.p_pos) 
		# 						for other in world.agents if other is not a]
		# 		if len(a_neighbor_dists) >= 2:
		# 			a_neighbor_dists.sort()
		# 			all_spacings.extend(a_neighbor_dists[:2])
			
		# 	spacing_std = np.std(all_spacings)
		# 	print("spacing_std",spacing_std)
		# 	rew -= spacing_std  # Reward uniform spacing across all agents
		# print("Agent",agent.id,"rew",rew)
		# input("Press Enter to continue...")

		return np.clip(rew, -2*self.collision_rew, self.goal_rew)



	def observation(self, agent: Agent, world: World) -> arr:
		"""
		Returns an observation for the agent, including:
		- Agent velocity
		- Agent position
		- Relative positions of the two nearest neighbors
		- Distances and occupancy status of the two closest goals
		- Tube-related information (distance to entrance/exit, width, and phase)
		"""
		agent_vel = agent.state.p_vel
		agent_pos = agent.state.p_pos
		
		# Find two nearest neighbors
		neighbor_dists = []
		for other in world.agents:
			if other is not agent:
				rel_pos = other.state.p_pos - agent.state.p_pos
				dist = np.linalg.norm(rel_pos)
				# print("dist",dist)
				# print("rel_pos",rel_pos)
				neighbor_dists.append((dist, rel_pos))
		
		# Sort by distance and get two nearest
		neighbor_dists.sort(key=lambda x: x[0])
		if len(neighbor_dists) >= 2:
			nearest_neighbors = [n[1] for n in neighbor_dists[:2]]
		else:
			nearest_neighbors = [n[1] for n in neighbor_dists]
			while len(nearest_neighbors) < 2:
				nearest_neighbors.append(np.zeros(world.dim_p))
		
		nearest_neighbors = np.concatenate(nearest_neighbors)
		# print("nearest_neighbors",nearest_neighbors)


		# === Find Two Closest Goals ===
		goal_dists = np.array([np.linalg.norm(agent.state.p_pos - l) for l in self.landmark_poses])
		sorted_goal_indices = np.argsort(goal_dists)  # Sort goal indices by distance
		top_two_indices = sorted_goal_indices[:2]  # Get the two closest goal indices
		min_dist = np.min(goal_dists)
		chosen_goal = top_two_indices[0]
		if min_dist < self.min_obs_dist:
			agents_goal = self.landmark_poses[top_two_indices[0]]
			# find which goals are within self.min_obs_dist and unoccupied
			nearby_goals = np.where(goal_dists < self.min_obs_dist)[0]
			# check if any of those have bee occupied and verify if true
			for goal in nearby_goals:
				# print("Agent",agent.id,"Nearby goal",goal,self.landmark_poses_occupied[goal])
				if self.landmark_poses_occupied[goal] == 1.0:
					# print("Agent",agent.id,"Nearby goal",goal, "is occupied")
					goal_proximity = np.array([np.linalg.norm(self.landmark_poses[goal] - agent.state.p_pos)  for agent in world.agents])
					if np.any(goal_proximity < self.min_dist_thresh):
						# print("Agent",agent.id,"Nearby goal",goal, "is occupied and another agent is at goal")
						continue
					else:
						# print("Agent",agent.id,"Nearby goal",goal, "is shown occupied but no agent is at goal")
						self.landmark_poses_occupied[goal] = 1 - np.min(goal_proximity)

			if min_dist < self.min_dist_thresh:
				if agent.status == True:
					self.landmark_poses_occupied[top_two_indices[0]] = 1.0
				else:
					self.landmark_poses_occupied[top_two_indices[0]] = 1.0-min_dist
				self.goal_history[top_two_indices[0]] = agent.id
				# print("Ag",agent.id," AT GOAL",np.min(world.dists), "goal_occupied",self.landmark_poses_occupied[chosen_goal])

			else:
				# goal_proximity is finding how many agents are near this chosen goal
				goal_proximity = np.array([np.linalg.norm(agents_goal - agent.state.p_pos)  for agent in world.agents])
				# print("Agent",agent.id,"chosen_goal", chosen_goal, "goal_proximity",goal_proximity, "flags",self.landmark_poses_occupied, "history",self.goal_history)
				closest_dist_to_goal = np.min(goal_proximity)


				# agent veered off the goal
				if self.landmark_poses_occupied[chosen_goal] == 1.0:

					# if there are no agents on the goal, then the agent can take the goal and change the occupancy value
					if np.any(goal_proximity < self.min_dist_thresh):
						# print("Agent!", "{:.0f}".format(self.goal_history[chosen_goal]), " is already at goal", "{:.0f}".format(chosen_goal), "min_dist", "{:.3f}".format(min_dist), "occupied flags",  self.landmark_poses_occupied, "history", self.goal_history)

						######
						## Add case when all nearby observed goals are occupied
						unoccupied_goals = self.landmark_poses[self.landmark_poses_occupied!= 1]
						unoccupied_goals_indices = np.where(self.landmark_poses_occupied != 1)[0]
						# print ("Agent",agent.id,"unoccupied_goals",unoccupied_goals, "unoccupied_goals_indices",unoccupied_goals_indices)
						assert len(unoccupied_goals) > 0, f"All goals are occupied {self.landmark_poses_occupied}, {self.goal_history},{world.dists} {goal_proximity}"
						# input("Press Enter to continue...")
						chosen_goal = np.argmin(np.linalg.norm(agent.state.p_pos - unoccupied_goals, axis=1))
						agents_goal = unoccupied_goals[chosen_goal]

					else:
						## add assertion to see if no goal is falsely occupied
						# assert not np.any(goal_proximity < self.min_dist_thresh), f"Agent {agent.id} is not at goal {chosen_goal} but flag is set to occupied"
						self.landmark_poses_occupied[chosen_goal] = 1.0-closest_dist_to_goal

				# another agent already at goal, can't overwrite the flag
				elif self.landmark_poses_occupied[chosen_goal] != 1.0:
					self.landmark_poses_occupied[chosen_goal] = 1.0-closest_dist_to_goal

			goal_occupied = np.array([self.landmark_poses_occupied[chosen_goal]])
			goal_history = self.goal_history[chosen_goal]

		else:
			# create another variable to store which goals are uncoccupied using an index of 0, 1 or 2 based on self.landmark_poses
			unoccupied_goals = self.landmark_poses[self.landmark_poses_occupied!= 1]

			unoccupied_goals_indices = np.where(self.landmark_poses_occupied != 1)[0]
			if len(unoccupied_goals) > 0:

				## determine which goal from self.landmark_poses is this chosen unocccupied goal
				## use the index of the unoccupied goal to get the goal from self.landmark_poses
				min_dist_goal = np.argmin(np.linalg.norm(agent.state.p_pos - unoccupied_goals, axis=1))
				agents_goal = unoccupied_goals[min_dist_goal]

				## check if the goal is occupied
				goal_occupied = np.array([self.landmark_poses_occupied[unoccupied_goals_indices[min_dist_goal]]])
				goal_history = self.goal_history[unoccupied_goals_indices[min_dist_goal]]

				## the second_closest_goal  needs to be from the unoccupied goals  or if there aren't any more unoccupied goals, it should be the closest from the occupied goals



			else:
				# Handle the case when all goals are occupied.
				agents_goal = agent.state.p_pos
				self.landmark_poses_occupied = np.zeros(self.num_agents)
				goal_history = self.goal_history[agent.id]
				goal_occupied = np.array([self.landmark_poses_occupied[agent.id]])
		


		# # Get goal positions and occupancy status

		second_closest_goal = self.landmark_poses[top_two_indices[1]]

		goal_pos = agents_goal - agent.state.p_pos
		rel_second_closest_goal = second_closest_goal - agent.state.p_pos

		goal_history = np.array([goal_history])
		closest_goal_occupied = goal_occupied ##np.array([self.landmark_poses_occupied[top_two_indices[0]]])
		# second_closest_goal_occupied = ## np.array([self.landmark_poses_occupied[top_two_indices[1]]])


		# Tube parameters
		tube_entrance = world.tube_params['entrance']
		tube_exit = world.tube_params['exit']
		tube_width = world.tube_params['width']
		
		# Calculate distances and directions to tube entrance/exit
		rel_to_entrance = tube_entrance - agent_pos
		rel_to_exit = tube_exit - agent_pos
		dist_to_entrance = np.linalg.norm(rel_to_entrance)
		dist_to_exit = np.linalg.norm(rel_to_exit)
		
		# Calculate phase
		in_tube = self.is_in_tube(world, agent_pos)
		passed_tube = agent_pos[1] < tube_exit[1]  # Assuming tube exit is at bottom
		
		if not in_tube and not passed_tube:
			phase = 0  # Pre-tube phase
		elif in_tube:
			phase = 1  # In-tube phase
		else:
			phase = 2  # Post-tube phase
		
		tube_params = np.concatenate([
			rel_to_entrance,  # Vector to tube entrance
			rel_to_exit,     # Vector to tube exit
			[tube_width],    # Tube width
			[phase]          # Current phase
		])
		# print("np.concatenate([	agent_vel,	agent_pos,	goal_pos,	closest_goal_occupied,rel_second_closest_goal,nearest_neighbors,tube_params)",
		# np.concatenate([	agent_vel,	agent_pos,	goal_pos,	closest_goal_occupied,rel_second_closest_goal,nearest_neighbors,tube_params]))

		return np.concatenate([
			agent_vel,
			agent_pos,
			goal_pos,
			closest_goal_occupied,
			rel_second_closest_goal,
			nearest_neighbors,
			tube_params
		])
	

	# def observation(self, agent:Agent, world:World) -> arr:
	# 	"""
	# 		Return:
	# 			[agent_vel, agent_pos, goal_pos, goal_occupied]
	# 	"""
	# 	# np.set_printoptions(formatter={'float': '{:0.3f}'.format})

	# 	world.dists = np.array([np.linalg.norm(agent.state.p_pos - l) for l in self.landmark_poses])
	# 	min_dist = np.min(world.dists)
	# 	sorted_indices = np.argsort(world.dists)  # Get indices that would sort the distances
	# 	top_two_indices = sorted_indices[:2]  # Get indices of top two closest distances
	# 	second_closest_goal = self.landmark_poses[top_two_indices[1]]
	# 	# get goal occupied flag for that goal
	# 	second_closest_goal_occupied = np.array([self.landmark_poses_occupied[top_two_indices[1]]])
	# 	if min_dist < self.min_obs_dist:
	# 		# If the minimum distance is already less than self.min_obs_dist, use the previous goal.
	# 		chosen_goal = np.argmin(world.dists)
	# 		agents_goal = self.landmark_poses[chosen_goal]


	# 		## check if any agent have left goals to go to other goals
	# 		# check which agents among the world.dists are less than self.min_obs_dist
	# 		# find which goals are within self.min_obs_dist and unoccupied
	# 		nearby_goals = np.where(world.dists < self.min_obs_dist)[0]
	# 		# check if any of those have bee occupied and verify if true
	# 		for goal in nearby_goals:
	# 			# print("Agent",agent.id,"Nearby goal",goal,self.landmark_poses_occupied[goal])
	# 			if self.landmark_poses_occupied[goal] == 1.0:
	# 				# print("Agent",agent.id,"Nearby goal",goal, "is occupied")
	# 				goal_proximity = np.array([np.linalg.norm(self.landmark_poses[goal] - agent.state.p_pos)  for agent in world.agents])
	# 				if np.any(goal_proximity < self.min_dist_thresh):
	# 					# print("Agent",agent.id,"Nearby goal",goal, "is occupied and another agent is at goal")
	# 					continue
	# 				else:
	# 					# print("Agent",agent.id,"Nearby goal",goal, "is shown occupied but no agent is at goal")
	# 					self.landmark_poses_occupied[goal] = np.min(goal_proximity)

	# 		if min_dist < self.min_dist_thresh:
	# 			if agent.status == True:
	# 				self.landmark_poses_occupied[chosen_goal] = 1.0
	# 			self.landmark_poses_occupied[chosen_goal] = 1.0-min_dist
	# 			self.goal_history[chosen_goal] = agent.id
	# 			# print("Ag",agent.id," AT GOAL",np.min(world.dists), "goal_occupied",self.landmark_poses_occupied[chosen_goal])

	# 		else:
	# 			# goal_proximity is finding how many agents are near this chosen goal
	# 			goal_proximity = np.array([np.linalg.norm(agents_goal - agent.state.p_pos)  for agent in world.agents])
	# 			# print("Agent",agent.id,"chosen_goal", chosen_goal, "goal_proximity",goal_proximity, "flags",self.landmark_poses_occupied, "history",self.goal_history)
	# 			closest_dist_to_goal = np.min(goal_proximity)


	# 			# agent veered off the goal
	# 			if self.landmark_poses_occupied[chosen_goal] == 1.0:

	# 				# if there are no agents on the goal, then the agent can take the goal and change the occupancy value
	# 				if np.any(goal_proximity < self.min_dist_thresh):
	# 					# print("Agent!", "{:.0f}".format(self.goal_history[chosen_goal]), " is already at goal", "{:.0f}".format(chosen_goal), "min_dist", "{:.3f}".format(min_dist), "occupied flags",  self.landmark_poses_occupied, "history", self.goal_history)

	# 					######
	# 					## Add case when all nearby observed goals are occupied
	# 					unoccupied_goals = self.landmark_poses[self.landmark_poses_occupied!= 1]
	# 					unoccupied_goals_indices = np.where(self.landmark_poses_occupied != 1)[0]
	# 					# print ("Agent",agent.id,"unoccupied_goals",unoccupied_goals, "unoccupied_goals_indices",unoccupied_goals_indices)
	# 					assert len(unoccupied_goals) > 0, f"All goals are occupied {self.landmark_poses_occupied}, {self.goal_history},{world.dists} {goal_proximity}"
	# 					# input("Press Enter to continue...")
	# 					chosen_goal = np.argmin(np.linalg.norm(agent.state.p_pos - unoccupied_goals, axis=1))
	# 					agents_goal = unoccupied_goals[chosen_goal]

	# 				else:
	# 					## add assertion to see if no goal is falsely occupied
	# 					# assert not np.any(goal_proximity < self.min_dist_thresh), f"Agent {agent.id} is not at goal {chosen_goal} but flag is set to occupied"
	# 					self.landmark_poses_occupied[chosen_goal] = 1.0-closest_dist_to_goal

	# 			# another agent already at goal, can't overwrite the flag
	# 			elif self.landmark_poses_occupied[chosen_goal] != 1.0:
	# 				self.landmark_poses_occupied[chosen_goal] = 1.0-closest_dist_to_goal

	# 		goal_occupied = np.array([self.landmark_poses_occupied[chosen_goal]])
	# 		goal_history = self.goal_history[chosen_goal]

	# 	else:
	# 		# create another variable to store which goals are uncoccupied using an index of 0, 1 or 2 based on self.landmark_poses
	# 		unoccupied_goals = self.landmark_poses[self.landmark_poses_occupied!= 1]

	# 		unoccupied_goals_indices = np.where(self.landmark_poses_occupied != 1)[0]
	# 		if len(unoccupied_goals) > 0:

	# 			## determine which goal from self.landmark_poses is this chosen unocccupied goal
	# 			## use the index of the unoccupied goal to get the goal from self.landmark_poses
	# 			min_dist_goal = np.argmin(np.linalg.norm(agent.state.p_pos - unoccupied_goals, axis=1))
	# 			agents_goal = unoccupied_goals[min_dist_goal]

	# 			## check if the goal is occupied
	# 			goal_occupied = np.array([self.landmark_poses_occupied[unoccupied_goals_indices[min_dist_goal]]])
	# 			goal_history = self.goal_history[unoccupied_goals_indices[min_dist_goal]]

	# 			## the second_closest_goal  needs to be from the unoccupied goals  or if there aren't any more unoccupied goals, it should be the closest from the occupied goals



	# 		else:
	# 			# Handle the case when all goals are occupied.
	# 			agents_goal = agent.state.p_pos
	# 			self.landmark_poses_occupied = np.zeros(self.num_agents)
	# 			goal_history = self.goal_history[agent.id]
	# 			goal_occupied = np.array([self.landmark_poses_occupied[agent.id]])
		
	# 	goal_pos = agents_goal - agent.state.p_pos
	# 	rel_second_closest_goal = second_closest_goal - agent.state.p_pos

	# 	goal_history = np.array([goal_history])

	# 	return np.concatenate((agent.state.p_vel, agent.state.p_pos, goal_pos,goal_occupied,goal_history, rel_second_closest_goal,second_closest_goal_occupied))

	def get_id(self, agent:Agent) -> arr:
		return np.array([agent.global_id])

	def collect_rewards(self,world):
		"""
		This function collects the rewards of all agents at once to reduce further computations of the reward
		input: world and agent information
		output: list of agent names and array of corresponding reward values [agent_names, agent_rewards]
		"""
		agent_names = []
		agent_rewards = np.zeros(self.num_agents)
		count = 0
		for agent in world.agents:
			agent_names.append(agent.name)
			agent_rewards[count] = self.reward(agent, world)
			count +=1
		return agent_names, agent_rewards

	
	def collect_dist(self, world):
		"""
		This function collects the distances of all agents at once to reduce further computations of the reward
		input: world and agent information
		output: mean distance, standard deviation of distance, and positions of agents
		"""
		agent_dist = np.array([agent.state.p_dist for agent in world.agents])  # Collect distances
		agent_pos = np.array([agent.state.p_pos for agent in world.agents])  # Collect positions

		mean_dist = np.mean(agent_dist)
		std_dev_dist = np.std(agent_dist)
		
		return mean_dist, std_dev_dist, agent_pos
	
	def sigmoid(x):
		return 1 / (1 + np.exp(-x))
	
	def collect_goal_info(self, world):
		goal_pos =  np.zeros((self.num_agents, 2)) # create a zero vector with the size of the number of goal and positions of dim 2
		count = 0
		for goal in world.landmarks:
			goal_pos[count]= goal.state.p_pos
			count +=1
		return goal_pos

	def graph_observation(self, agent:Agent, world:World) -> Tuple[arr, arr]:
		"""
			FIXME: Take care of the case where edge_list is empty
			Returns: [node features, adjacency matrix]
			• Node features (num_entities, num_node_feats):
				If `global`: 
					• node features are global [pos, vel, goal, entity-type]
					• edge features are relative distances (just magnitude)
					NOTE: for `landmarks` and `obstacles` the `goal` is 
							the same as its position
				If `relative`:
					• node features are relative [pos, vel, goal, entity-type] to ego agents
					• edge features are relative distances (just magnitude)
					NOTE: for `landmarks` and `obstacles` the `goal` is 
							the same as its position
			• Adjacency Matrix (num_entities, num_entities)
				NOTE: using the distance matrix, need to do some post-processing
				If `global`:
					• All close-by entities are connectd together
				If `relative`:
					• Only entities close to the ego-agent are connected
			
		"""

		# node observations
		node_obs = []
		fairness_param = 0.0

		if world.graph_feat_type == 'global':
			for i, entity in enumerate(world.entities):

				node_obs_i = self._get_entity_feat_global(entity, world)
				node_obs.append(node_obs_i)

		elif world.graph_feat_type == 'relative':
			for i, entity in enumerate(world.entities):

				node_obs_i = self._get_entity_feat_relative(agent, entity, world, fairness_param)
				node_obs.append(node_obs_i)

		node_obs = np.array(node_obs)
		adj = world.cached_dist_mag

		return node_obs, adj

	def update_graph(self, world:World):
		"""
			Construct a graph from the cached distances.
			Nodes are entities in the environment
			Edges are constructed by thresholding distances
		"""
		dists = world.cached_dist_mag
		# just connect the ones which are within connection 
		# distance and do not connect to itself
		connect = np.array((dists <= self.max_edge_dist) * \
							(dists > 0)).astype(int)
		sparse_connect = sparse.csr_matrix(connect)
		sparse_connect = sparse_connect.tocoo()
		row, col = sparse_connect.row, sparse_connect.col
		edge_list = np.stack([row, col])
		world.edge_list = edge_list
		if world.graph_feat_type == 'global':
			world.edge_weight = dists[row, col]
		elif world.graph_feat_type == 'relative':
			world.edge_weight = dists[row, col]
	
	def _get_entity_feat_global(self, entity:Entity, world:World) -> arr:
		"""
			Returns: ([velocity, position, goal_pos, entity_type])
			in global coords for the given entity
		"""
		pos = entity.state.p_pos
		vel = entity.state.p_vel
		if 'agent' in entity.name:
			goal_pos = world.get_entity('landmark', self.goal_match_index[entity.id]).state.p_pos
			entity_type = entity_mapping['agent']
		elif 'landmark' in entity.name:
			goal_pos = pos
			entity_type = entity_mapping['landmark']
		elif 'obstacle' in entity.name:
			goal_pos = pos
			entity_type = entity_mapping['obstacle']
		else:
			raise ValueError(f'{entity.name} not supported')

		return np.hstack([vel, pos, goal_pos, entity_type])

	def _get_entity_feat_relative(self, agent:Agent, entity:Entity, world:World, fairness_param: np.ndarray) -> arr:
		"""
			Returns: ([velocity, position, goal_pos, entity_type])
			in coords relative to the `agent` for the given entity
		"""
		agent_pos = agent.state.p_pos
		agent_vel = agent.state.p_vel
		entity_pos = entity.state.p_pos
		entity_vel = entity.state.p_vel
		rel_pos = entity_pos - agent_pos
		rel_vel = entity_vel - agent_vel
		if 'agent' in entity.name:
			world.dists = np.array([np.linalg.norm(entity.state.p_pos - l) for l in self.landmark_poses])
			min_dist = np.min(world.dists)
			if min_dist < self.min_obs_dist:
				# If the minimum distance is already less than self.min_dist_thresh, use the previous goal.
				chosen_goal = np.argmin(world.dists)
				goal_pos = self.landmark_poses[chosen_goal]
				goal_history = self.goal_history[chosen_goal]
				goal_occupied = np.array([self.landmark_poses_occupied[chosen_goal]])

			else:
				unoccupied_goals = self.landmark_poses[self.landmark_poses_occupied!= 1]
				unoccupied_goals_indices = np.where(self.landmark_poses_occupied != 1)[0]
				if len(unoccupied_goals) > 0:

					## use closest goal

					## determine which goal from self.landmark_poses is this chosen unocccupied goal
					## use the index of the unoccupied goal to get the goal from self.landmark_poses
					min_dist_goal = np.argmin(np.linalg.norm(entity.state.p_pos - unoccupied_goals, axis=1))
					goal_pos = unoccupied_goals[min_dist_goal]
					## check if the goal is occupied
					goal_occupied = np.array([self.landmark_poses_occupied[unoccupied_goals_indices[min_dist_goal]]])
					goal_history = self.goal_history[unoccupied_goals_indices[min_dist_goal]]

				else:
					# Handle the case when all goals are occupied.
					goal_pos = entity.state.p_pos
					self.landmark_poses_occupied = np.zeros(self.num_agents)
					goal_occupied = np.array([self.landmark_poses_occupied[entity.id]])
					goal_history = self.goal_history[entity.id]

			goal_history = np.array([goal_history])

			rel_goal_pos = goal_pos - agent_pos
			entity_type = entity_mapping['agent']


		elif 'landmark' in entity.name:
			rel_goal_pos = rel_pos
			goal_occupied = np.array([1])
			goal_history = entity.id if entity.id != None else 0
			entity_type = entity_mapping['landmark']

		elif 'obstacle' in entity.name:
			rel_goal_pos = rel_pos
			goal_occupied = np.array([1])
			goal_history = entity.id if entity.id != None else 0
			entity_type = entity_mapping['obstacle']

		elif 'wall' in entity.name:
			rel_goal_pos = rel_pos
			goal_occupied = np.array([1])
			goal_history = entity.id if entity.id != None else 0
			entity_type = entity_mapping['wall']
			## get wall corner point's relative position
			wall_o_corner = np.array([entity.endpoints[0],entity.axis_pos+entity.width/2]) - agent_pos
			wall_d_corner = np.array([entity.endpoints[1],entity.axis_pos-entity.width/2]) - agent_pos
			return np.hstack([rel_vel, rel_pos, rel_goal_pos,goal_occupied,goal_history,wall_o_corner,wall_d_corner,entity_type])

		else:
			raise ValueError(f'{entity.name} not supported')
		return np.hstack([rel_vel, rel_pos, rel_goal_pos,goal_occupied,goal_history,rel_pos,rel_pos,entity_type])



# actions: [None, ←, →, ↓, ↑, comm1, comm2]
if __name__ == "__main__":

	from multiagent.environment import MultiAgentGraphEnv
	from multiagent.policy import InteractivePolicy

	# makeshift argparser
	class Args:
		def __init__(self):
			self.num_agents:int=3
			self.world_size=2
			self.num_scripted_agents=0
			self.num_obstacles:int=3
			self.collaborative:bool=False 
			self.max_speed:Optional[float]=2
			self.collision_rew:float=5
			self.goal_rew:float=50
			self.min_dist_thresh:float=0.1
			self.use_dones:bool=True
			self.episode_length:int=25
			self.max_edge_dist:float=1
			self.graph_feat_type:str='relative'
			self.fair_wt=1
			self.fair_rew=1
	args = Args()

	scenario = Scenario()
	# create world
	world = scenario.make_world(args)
	# create multiagent environment
	env = MultiAgentGraphEnv(world=world, reset_callback=scenario.reset_world, 
						reward_callback=scenario.reward, 
						observation_callback=scenario.observation, 
						graph_observation_callback=scenario.graph_observation,
						info_callback=scenario.info_callback, 
						done_callback=scenario.done,
						id_callback=scenario.get_id,
						update_graph=scenario.update_graph,
						shared_viewer=False)
	# render call to create viewer window
	env.render()
	# create interactive policies for each agent
	policies = [InteractivePolicy(env,i) for i in range(env.n)]
	# execution loop
	obs_n, agent_id_n, node_obs_n, adj_n = env.reset()
	stp=0

	prev_rewards = []
	while True:
		# query for action from each agent's policy
		act_n = []
		dist_mag = env.world.cached_dist_mag

		for i, policy in enumerate(policies):
			act_n.append(policy.action(obs_n[i]))
		# step environment

		obs_n, agent_id_n, node_obs_n, adj_n, reward_n, done_n, info_n = env.step(act_n)
		prev_rewards= reward_n

		# render all agent views
		env.render()
		stp+=1
		# display rewards
