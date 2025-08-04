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

# import numba
# from numba import cuda, jit

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

# @staticmethod
# @jit(nopython=True)
def get_rotated_position_from_relative(relative_position: np.ndarray,
	reference_heading: float) -> np.ndarray:
	# returns relative position from the reference state.
	assert relative_position.shape == (2,), "relative_position should be a 2D array."
	rot_matrix = np.array([[np.cos(reference_heading), np.sin(reference_heading)], [-np.sin(reference_heading), np.cos(reference_heading)]])
	relative_position_rotated = np.dot(rot_matrix, relative_position)
	return relative_position_rotated

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
		self.args = args
		self.num_agents = args.num_agents
		self.num_scripted_agents = args.num_scripted_agents
		self.num_obstacles = args.num_obstacles
		self.collaborative = args.collaborative
		self.max_speed = args.max_speed
		self.collision_rew = args.collision_rew
		self.formation_rew = args.formation_rew
		self.goal_rew = args.goal_rew

		# dummy_pos = np.array([1.0, 0.0])
		# get_rotated_position_from_relative(dummy_pos, 0.0)
		self.use_dones = args.use_dones
		self.episode_length = args.episode_length
		# used for curriculum learning (in training)
		self.num_total_episode = int(args.num_env_steps) // args.episode_length // args.n_rollout_threads
		if args.render_episodes is not None:
			self.num_total_episode = args.render_episodes

		self.target_radius = 0.5  # fixing the target radius for now
		self.ideal_theta_separation = (
			2 * np.pi
		) / self.num_agents  # ideal theta difference between two agents

		# fairness args
		self.fair_wt = args.fair_wt
		self.fair_rew = args.fair_rew

		self.formation_type = args.formation_type
		self.steps_in_corridor = np.zeros(self.num_agents)
		# create heatmap matrix to determine the goal agent pairs
		self.goal_reached = np.full(self.num_agents, -1)
		self.wrong_goal_reached = np.zeros(self.num_agents)
		self.goal_matched = np.zeros(self.num_agents)

		self.goal_tracker = np.full(self.num_agents, -1)## 	keeps track of which goal each agent goes to using self.goal_tracker[agent.id] = self.goal_match_index[agent.id]

		self.conformance_percent = np.zeros(self.num_agents)

		self.delta_spacing = []

		self.current_tube =np.zeros(self.num_agents, dtype=int)
		self.spacing_violation = np.zeros(self.num_agents)

		self.max_tubes = 3

		self.tube_choice = 0
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

		self.phase_reached = np.zeros(self.num_agents)  ## keeps track of which phase each agent is in when it first enters it
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
		# needed when scenario is used in evaluation.
		self.curriculum_ratio = 1.0
		self.max_edge_dist = self.coordination_range
		self.goal_match_index = np.zeros([self.max_tubes,self.num_agents], dtype=int)
		self.goal_match_index[0]	= np.arange(self.num_agents, dtype=int)
		self.goal_match_index[1]	= np.arange(self.num_agents,	 dtype=int)
		self.goal_match_index[2]	= np.arange(self.num_agents, dtype=int)
		# print("GOAL MATCH INDEX",self.goal_match_index)
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
		world.times_required =  np.full(self.num_agents, -1)
		world.dists_to_goal =  np.full(self.num_agents, -1)
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

	def reset_world(self, world:World, num_current_episode: int = 0) -> None:
		# print("RESET WORLD")
		# metrics to keep track of
		world.current_time_step = 0
		world.simulation_time = 0.0
		# to track time required to reach goal
		world.times_required = np.full(self.num_agents, -1)
		world.dists_to_goal =  np.full(self.num_agents, -1)

		# track distance left to the goal
		world.dist_left_to_goal =  np.full(self.num_agents, -1)
		# number of times agents collide with stuff
		world.num_obstacle_collisions = np.zeros(self.num_agents)
		world.num_goal_collisions = np.zeros(self.num_agents)
		world.num_agent_collisions = np.zeros(self.num_agents)
		world.agent_dist_traveled = np.zeros(self.num_agents)

		self.goal_match_index = np.zeros([self.max_tubes,self.num_agents], dtype=int)
		self.goal_match_index[0]	= np.arange(self.num_agents, dtype=int)
		self.goal_match_index[1]	= np.arange(self.num_agents,	 dtype=int)
		self.goal_match_index[2]	= np.arange(self.num_agents, dtype=int)
		self.goal_history = np.full(self.num_agents, -1)
		self.goal_reached =  np.full(self.num_agents, -1)

		self.goal_tracker = np.full(self.num_agents, -1)
		self.conformance_percent = np.zeros(self.num_agents)
		self.delta_spacing =[]
		self.spacing_violation = np.zeros(self.num_agents)
		self.steps_in_corridor = np.zeros(self.num_agents)

		self.agent_dist_traveled = np.zeros(self.num_agents)
		self.agent_time_taken = np.zeros(self.num_agents)
		wall_length = np.random.uniform(0.2, 0.8)
		self.wall_length = wall_length * self.world_size/4

		self.phase_reached = np.zeros(self.num_agents)

		self.current_tube =  np.zeros(self.num_agents, dtype=int)
		self.tube_choice = 0

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
		# self.update_curriculum(world, num_current_episode)
		self.random_scenario(world)
		self.initialize_min_time_distance_graph(world)


	
	def update_curriculum(self, world:World, num_current_episode:int) -> None:

		""" Update the curriculum learning parameters if necessary."""
		# print(f"Current Episode: {num_current_episode}")
		# print(f"Total Episodes: {self.num_total_episode}")
		self.curriculum_ratio = np.clip(num_current_episode / self.num_total_episode, 0.1, 1.0)
		# print(f"Curriculum Ratio: {self.curriculum_ratio}")
		## update collision penalty
		self.collision_rew = self.args.collision_rew * self.curriculum_ratio
		# print(f"Collision Reward: {self.collision_rew}")
		## update formation reward
		self.formation_rew = self.args.formation_rew * self.curriculum_ratio

		## update fairness reward
		self.fair_rew = self.args.fair_rew * self.curriculum_ratio

	def random_scenario(self, world):
		"""
			Randomly place agents and landmarks
		"""

		# set agents at random positions not colliding with obstacles
		num_agents_added = 0
		agents_added = []
		boundary_thresh = 0.9

		while True:
			if num_agents_added == self.num_agents:
				break
			# Random position in top half of world
			random_pos = boundary_thresh * np.random.uniform(
				[-self.world_size,self.world_size/2],  # min x,y
				[self.world_size, self.world_size],  # max x,y
				world.dim_p
			)

			# ## shift agents to bottom half of the world
			# random_pos[1] = random_pos[1] - self.world_size/2
			agent_size = world.agents[num_agents_added].size
			obs_collision = self.is_obstacle_collision(random_pos, agent_size, world)
			# goal_collision = self.is_goal_collision(uniform_pos, agent_size, world)

			agent_collision = self.check_agent_collision(random_pos, agent_size, agents_added)
			if not obs_collision and not agent_collision:
				world.agents[num_agents_added].state.p_pos = random_pos
				world.agents[num_agents_added].state.reset_velocity()
				world.agents[num_agents_added].state.c = np.zeros(world.dim_c)
				world.agents[num_agents_added].status = False
				agents_added.append(world.agents[num_agents_added])
				num_agents_added += 1
		agent_pos = [agent.state.p_pos for agent in world.agents]
		#####################################################
		# Initialize tube parameters
		self.setup_tube_params(world)

		self.agent_id_updated = np.arange(self.num_agents)
		if self.formation_type == 'line':
			set_landmarks_in_line(self, world, line_angle=0, start_pos=np.array([-self.world_size/2, -self.world_size/2]), end_pos=np.array([self.world_size/2,-self.world_size/2]))
		elif self.formation_type == 'circle':
			set_landmarks_in_circle(self, world, center=np.array([0.0, world.tube_params['exit'][1]+self.world_size/5]), radius=self.world_size/3)
		elif self.formation_type == 'point':
			set_landmarks_in_point(self, world, tube_endpoints=world.tube_params[0]['exit'])
		# elif self.formation_type == 'random':
		# 	set_landmarks_random(self, world)
		# else:
		# 	raise NotImplementedError

		# Update landmark poses arrays
		self.landmark_poses = np.array([landmark.state.p_pos for landmark in world.landmarks])
		# print("LANDMARK POSES",self.landmark_poses)
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


	# def setup_tube_params(self, world):
	# 	"""
	# 	Set up tube parameters using modified landmark line logic
	# 	"""
	# 	# Calculate tube width based on number of agents
	# 	self.tube_width = max(
	# 		3 * world.agents[0].size * 2.5,  # Width based on agents # =3  TODO: harcoded
	# 		self.world_size * 0.15  # Minimum width
	# 	)
		
	# 	# Calculate tube entrance (middle of world)
	# 	entrance_y = self.world_size/2 * 0.5  # Middle of world
	# 	entrance_x = 0  # Center horizontally
		
	# 	# Calculate tube exit (bottom of world)
	# 	exit_y = -self.world_size/2 * 0.5  # Near bottom of world
	# 	exit_x = 0  # Same x as entrance
		
	# 	# Store tube parameters
	# 	world.tube_params = {
	# 		'entrance': np.array([entrance_x, entrance_y]),
	# 		'exit': np.array([exit_x, exit_y]),
	# 		'width': self.tube_width,
	# 		'angle': np.pi/2  # Vertical tube
	# 	}

	# 	# Calculate line formation reference points
	# 	line_length = self.tube_width * 0.8  # Slightly smaller than tube width
		
	# 	# Pre-tube formation line (above entrance)
	# 	pre_tube_y = entrance_y + self.world_size * 0.2
	# 	self.pre_tube_line = {
	# 		'start_pos': np.array([-line_length/2, pre_tube_y]),
	# 		'end_pos': np.array([line_length/2, pre_tube_y]),
	# 		'angle': 0  # Horizontal line
	# 	}
		
	# 	# Post-tube target line (at top of world)
	# 	post_tube_y = self.world_size/2 * 0.8
	# 	self.post_tube_line = {
	# 		'start_pos': np.array([-line_length/2, post_tube_y]),
	# 		'end_pos': np.array([line_length/2, post_tube_y]),
	# 		'angle': 0  # Horizontal line
	# 	}
		
	# 	# Store additional parameters that might be useful
	# 	world.tube_params.update({
	# 		'length': entrance_y - exit_y,
	# 		'pre_tube_line': self.pre_tube_line,
	# 		'post_tube_line': self.post_tube_line
	# 	})

	def setup_tube_params(self, world):
		"""
		Set up two perpendicular tubes using modified landmark line logic.
		"""

		# Initialize tube list
		world.tube_params = []

		# Calculate tube width based on agents
		tube_width = max(
			3 * world.agents[0].size * 2.5,  # Width based on agents
			self.world_size * 0.15  # Minimum width
		)

		# First tube (Vertical)
		entrance1 = np.array([0, self.world_size / 2 * 0.5])  # Middle top
		exit1 = np.array([0, -self.world_size / 2 * 0.5])  # Near bottom
		tube1 = {
			'entrance': entrance1,
			'exit': exit1,
			'width': tube_width,
			'angle': np.pi / 2,  # Vertical tube
			'length': entrance1[1] - exit1[1]
		}
		
		# Store first tube
		world.tube_params.append(tube1)
		# print("TUBE1",tube1)

		# # ## RANDMIZE DIRECTION OF SECOND TUBE
		# if np.random.choice([0,1]) == 1:
		# self.tube_choice = 1

		# Second tube (Perpendicular to exit of first tube)
		entrance2 = exit1 + np.array([+self.world_size/5, -self.world_size/5])  # Entrance of tube2 is exit of tube1
		exit2 = np.array([self.world_size * 0.5, entrance2[1]])  # Moves right from the exit of tube1

		tube2 = {
			'entrance': entrance2,
			'exit': exit2,
			'width': tube_width,
			'angle': 0,  # Horizontal tube
			'length': np.linalg.norm(exit2 - entrance2)  # Euclidean distance
		}
		# print("TUBE2",tube2)
		# Store second tube
		world.tube_params.append(tube2)
		# else:
		# Second tube (Perpendicular to exit of first tube)
		# self.tube_choice = 0
		entrance2 = exit1 + np.array([-self.world_size/5, -self.world_size/5])
		exit2 = np.array([-self.world_size * 0.5, entrance2[1]])  # Moves left from the exit of tube1
		# print("ENTRANCE2",entrance2)
		# print("EXIT2",exit2)
		tube3 = {
			'entrance': entrance2,
			'exit': exit2,
			'width': tube_width,
			'angle': 0,  # Horizontal tube
			'length': np.linalg.norm(exit2 - entrance2)  # Euclidean distance
		}
		# print("TUBE2",tube2)
		# Store second tube
		world.tube_params.append(tube3)

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

		# print("CONFORMANCE",self.conformance_percent)
		# print("self.conformance_percent[agent.id]/self.args.episode_length,",self.conformance_percent[agent.id]/self.args.episode_length)
		# print("SPACING VIOLATION",self.spacing_violation)
		# print("DELTA SPACING",self.delta_spacing)
		# convert delta spacing to an array
		self.delta_spacing_sum = np.array(self.delta_spacing)
		#sum all the values in the list
		self.delta_spacing_sum = np.sum(self.delta_spacing, axis=0)
		# print("self.delta_spacing_sum",self.delta_spacing_sum)
		# print("self.spacing_violation",np.sum(self.spacing_violation))
		# print(self.delta_spacing[agent.id]/(self.spacing_violation[agent.id] if self.spacing_violation[agent.id] != 0 else 1))
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
			'Conformance': self.conformance_percent[agent.id]/self.args.episode_length,
			'Delta_spacing': self.delta_spacing_sum /(np.sum(self.spacing_violation) if np.sum(self.spacing_violation) != 0 else 1),
			'Spacing_violations': self.spacing_violation[agent.id]/(self.steps_in_corridor[agent.id] if self.steps_in_corridor[agent.id] != 0 else 1),


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
				if dist < self.separation_distance:
					collision = True
					break
		return collision

	# check collision of agent with another agent
	def is_collision(self, agent1:Agent, agent2:Agent) -> bool:
		if agent1.status or agent2.status:
			# print("agent status",agent1.status,agent2.status)
			return False
		delta_pos = agent1.state.p_pos - agent2.state.p_pos
		dist = np.linalg.norm(delta_pos)
		dist_min = self.separation_distance
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
		# print("self.goal_match_index",self.goal_match_index)
		# print("self.current_tube",self.current_tube, "agent_id",agent_id)
		landmark = world.get_entity(entity_type='landmark', id=self.goal_match_index[self.current_tube[agent.id],agent.id])
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
				landmark = world.get_entity('landmark',self.goal_match_index[self.current_tube[agent.id],agent.id])
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


		# Common rewards across all phases
		# Collision penalties
		if agent.collide:
			for a in world.agents:
				if a.id == agent.id:
					continue
				if self.is_collision(a, agent):
					rew -= self.collision_rew*4
					# print(f"Agent {agent.id} collided with agent {a.id} penalty",self.collision_rew*3 )
					# input("Collision")
			
			if self.is_obstacle_collision(pos=agent.state.p_pos,
										entity_size=agent.size, 
										world=world):
				rew -= self.collision_rew*3
				# print(f"Agent {agent.id} collided with obstacle")

		# Calculate tube length
		tube_length = world.tube_params[self.current_tube[agent.id]]['entrance'][1] - world.tube_params[self.current_tube[agent.id]]['exit'][1]
		
		# Find agents in front and behind
		agents_sorted_by_y = sorted(world.agents, key=lambda a: a.state.p_pos[1], reverse=True)
		agent_idx = agents_sorted_by_y.index(agent)
		# print("Agent",agent.id,"Current Phase",current_phase, "Phase_reached",self.phase_reached)

		# Get closest agent in front and behind
		back_agent = agents_sorted_by_y[agent_idx - 1] if agent_idx > 0 else None
		# print("back_agent",back_agent.id if back_agent else "None")
		front_agent = agents_sorted_by_y[agent_idx + 1] if agent_idx < len(agents_sorted_by_y) - 1 else None
		# print("front_agent",front_agent.id if front_agent else "None")
		
		# Calculate desired spacing based on tube length and number of agents
		desired_spacing = tube_length / (4 + 1) # 3 is the number of agents in the tube TODO: harcoded

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
			# print("self.goal_rew",self.goal_rew)
			rew -= self.goal_rew*3  # Reduced penalty
			# print(f"Agent {agent.id} penalized for skipping from phase {agent.previous_phase} to {current_phase} rew", rew)
		if current_phase == agent.previous_phase+1 and self.phase_reached[agent.id] == current_phase-1:
			# Reward proper phase progression
			if current_phase == 1 and agent.state.p_pos[1] >= (world.tube_params[self.current_tube[agent.id]]['entrance'][1]-0.2*tube_length):
				# print("Agent pos, entrance, exit",agent.state.p_pos[1],(world.tube_params['entrance'][1]+world.tube_params['exit'][1])/2)
				# Reward if agent moves into tube after exiting
				rew += self.goal_rew*3  # Positive reward for proper transition
				# print(f"Agent {agent.id} properly progressed from phase {agent.previous_phase} to {current_phase} rew", rew)
			elif current_phase == 2 :
				# Rewards if agent moves out of tube
				rew += self.goal_rew*3
				# print(f"Agent {agent.id} properly progressed from phase {agent.previous_phase} to {current_phase} rew", rew)
				# Update the global phase tracker if any agent progresses

		# Phase-specific rewards
		# print("Agent",agent.id,"current", current_phase,"Phase_reached till now",self.phase_reached)
		if current_phase == 0:  # Pre-tube phase
			# Reward for getting closer to tube entrance
			dist_to_entrance = np.linalg.norm(world.tube_params[self.current_tube[agent.id]]['entrance'] - agent.state.p_pos)
			rew -= dist_to_entrance
			# if self.current_tube[agent.id] >0:
			# 	print("Phase 0dist_to_entrance",dist_to_entrance, "rew", rew)
			# 	input("phase 0")
			# Formation reward considering front and back agents
			spacing_error = 0
			if front_agent:
				diff = np.linalg.norm(front_agent.state.p_pos - agent.state.p_pos) - desired_spacing
				spacing_error += np.abs(diff) if diff < 0 else 0
			if back_agent:
				diff = np.linalg.norm(back_agent.state.p_pos - agent.state.p_pos) - desired_spacing
				spacing_error += np.abs(diff) if diff < 0 else 0
			rew -= spacing_error *  self.formation_rew
			# print("Phase 0 spacing_error",spacing_error)
				
		elif current_phase == 1:  # In-tube phase
			# print("formation line",self.formation_rew)
			# rew += self.formation_rew/2  # Reward for entering tube
			# Stronger formation rewards inside tube
			spacing_error = 0
			max_spacing_error = 0
			if front_agent:
				diff = np.linalg.norm(front_agent.state.p_pos - agent.state.p_pos) - desired_spacing
				spacing_error += np.abs(diff) if diff < 0 else 0
				max_spacing_error = max(max_spacing_error, np.abs(diff))
			if back_agent:
				diff = np.linalg.norm(back_agent.state.p_pos - agent.state.p_pos) - desired_spacing
				max_spacing_error = max(max_spacing_error, np.abs(diff))
				spacing_error += np.abs(diff) if diff < 0 else 0
			if spacing_error > 0:
				self.spacing_violation[agent.id] += 1
			rew -= spacing_error *  self.formation_rew # Higher weight for maintaining formation in tube
			# print("Phase 1 spacing_error",spacing_error)
			# Progress through tube
			dist_to_exit = np.linalg.norm(world.tube_params[self.current_tube[agent.id]]['exit'] - agent.state.p_pos)
			rew -= dist_to_exit
			# print("dist_to_exit",dist_to_exit, "rew", rew)
			self.delta_spacing.append(spacing_error)
			self.steps_in_corridor[agent.id] += 1
			# print("Agent",agent.id,"delta_spacing",self.delta_spacing[agent.id])
			# if self.current_tube[agent.id]==1:
			# 	# print("Agent", agent.id, "dist_to_exit",dist_to_exit, "rew", rew)
			# 	pos = agent.state.p_pos
			# 	tube = world.tube_params[self.current_tube[agent.id]]
			# 	in_tube = self.is_in_tube(world, pos, agent)
				# print("Is agent in tube?",in_tube)

			# input("phase 1")
	
		elif current_phase == 2 and self.phase_reached[agent.id] == 0:  # Post-tube phase
			# print("Agent",agent.id,"post tube phase", self.current_tube[agent.id])
			# input("Agent entered post tube phase")
			# input("Agent entered post tube phase")
			current_phase = 0  # Reset current phase to 0
		elif current_phase == 2 and self.current_tube[agent.id] == 0:
				# print("Agent", agent.id,"moving to next tube")
				if agent.id%3 == 0:
					adding = 1
				else:
					adding = 0
				self.current_tube[agent.id] = 1+adding#np.random.choice([0,1])
				# print("Agent", agent.id,"new tube", self.current_tube[agent.id])
				self.phase_reached[agent.id] = 0
				current_phase = 0
				agent.previous_phase = 0
				set_landmarks_in_point_seq(self, world, tube_endpoints=world.tube_params[self.current_tube[agent.id]]['entrance'], agent_id=agent.id,tube_choice = self.current_tube[agent.id] )
				# pos = agent.state.p_pos
				# tube = world.tube_params[self.current_tube[agent.id]]
				# in_tube = self.is_in_tube(world, pos, agent)
				# print("Is agent in tube?",in_tube)	
				# print("tube['angle']",tube['angle'])
				# ##If tube is vertical, check Y-coordinate for "passed_tube"

				# input("tube 2")
		else:
			dist_to_goal = np.linalg.norm(agent.state.p_pos - self.landmark_poses[self.goal_match_index[self.current_tube[agent.id],agent.id]])
			if dist_to_goal < self.min_dist_thresh:
				# print("Agent",agent.id,"reached fair goal")
				if agent.status ==False:
					agent.status = True
					agent.state.reset_velocity()
					rew += self.goal_rew*5
					self.goal_tracker[agent.id] = self.goal_match_index[self.current_tube[agent.id],agent.id]

					# print("Phase 2 Agent",agent.id,"reached goal")

			else:
					# print("dist_to_goal",dist_to_goal, "rew", rew)
					rew -= dist_to_goal
					# input("phase 2")

			spacing_error = 0
			if front_agent:
				diff = np.linalg.norm(front_agent.state.p_pos - agent.state.p_pos) - desired_spacing
				spacing_error += np.abs(diff) if diff < 0 else 0
			if back_agent:
				diff = np.linalg.norm(back_agent.state.p_pos - agent.state.p_pos) - desired_spacing
				spacing_error += np.abs(diff) if diff < 0 else 0
			rew -= spacing_error *  self.formation_rew # Higher weight for maintaining formation in tube

		# # Global formation quality (calculated once per step)
		if agent.id == 0:
			all_spacings = []
			for a in world.agents:
				a_neighbor_dists = [np.linalg.norm(other.state.p_pos - a.state.p_pos) 
								for other in world.agents if other is not a]
				if len(a_neighbor_dists) >= 2:
					a_neighbor_dists.sort()
					all_spacings.extend(a_neighbor_dists[:2])
			# print("all_spacings",all_spacings)
			spacing_std = np.std(all_spacings)
			# self.spacing_min = np.min(all_spacings)
			# print("Global spacing_std",spacing_std*self.formation_rew )
			rew -= spacing_std *  self.formation_rew  # Reward uniform spacing across all agents
		# print("Agent",agent.id,"rew",rew)



		# print("Agent.status",agent.status)
		if self.phase_reached[agent.id] == 1 and current_phase == 0:
			# print("Agent",agent.id,"left corridor")
			# input("Agent left corridor")
			self.conformance_percent[agent.id] += 1
			# print("conformance_percent",self.conformance_percent[agent.id])

		if current_phase > self.phase_reached[agent.id]:
			# print(f"Agent {agent.id} reached phase {current_phase}")
			self.phase_reached[agent.id] = current_phase  # Update max phase reached globally
		## penalize for moving from higher phase to lower phase
		if current_phase < agent.previous_phase:
			rew -= self.collision_rew*3
			# print(f"Agent {agent.id} tried to move back to phase {current_phase} from {agent.previous_phase} rew", rew)
		if current_phase < self.phase_reached[agent.id]:
			rew -= self.collision_rew
			# print(f"Agent {agent.id} tried to move back to phase {current_phase} from {self.phase_reached[agent.id]} rew", rew)
		# Store current phase for next step
		agent.previous_phase = current_phase
		return np.clip(rew, -4*self.collision_rew, self.goal_rew*5)

	# def is_in_tube(self, world: World, pos):
	# 	"""Helper function to check if position is inside tube"""
	# 	tube_entrance = world.tube_params[self.current_tube]['entrance']
	# 	tube_exit = world.tube_params[self.current_tube]['exit']
	# 	tube_width = world.tube_params[self.current_tube]['width']
	# 	print("current_tube",self.current_tube)
	# 	# Check if position is between entrance and exit y-coordinates
	# 	in_y_range = pos[1] <= tube_entrance[1] and pos[1] >= tube_exit[1]
		
	# 	# Check if position is within tube width
	# 	in_x_range = abs(pos[0] - tube_entrance[0]) <= tube_width/2
		
	# 	return in_y_range and in_x_range

	def is_in_tube(self, world: World, pos, agent:Agent):
		"""Helper function to check if position is inside tube (supports both vertical and horizontal tubes)"""
		
		tube = world.tube_params[self.current_tube[agent.id]]  # Get the current tube
		tube_entrance = tube['entrance']
		tube_exit = tube['exit']
		tube_width = tube['width']
		tube_angle = tube['angle']  # Determines tube orientation
		
		# print("current_tube:", self.current_tube)

		# If tube is vertical (angle ≈ π/2 or -π/2)
		if np.isclose(abs(tube_angle), np.pi / 2, atol=1e-2):
			in_y_range = tube_exit[1] <= pos[1] <= tube_entrance[1]  # Check Y-range
			in_x_range = abs(pos[0] - tube_entrance[0]) <= tube_width / 2  # Check X-width

		# If tube is horizontal (angle ≈ 0 or π)
		elif np.isclose(tube_angle, 0, atol=1e-2) or np.isclose(abs(tube_angle), np.pi, atol=1e-2):
			# x coordinate should be within the tube entrance and exit range, even if entrance and exit are negative and reversed
			in_x_range = min(tube_entrance[0], tube_exit[0]) <= pos[0] <= max(tube_entrance[0], tube_exit[0])
			# in_x_range = tube_entrance[0] <= pos[0] <= tube_exit[0]  # Check X-range
			in_y_range = abs(pos[1] - tube_entrance[1]) <= tube_width / 2  # Check Y-width
			# print("is in tube?", in_y_range, in_x_range, pos, tube_entrance, tube_exit)

		else:
			# Handle diagonal tubes (optional, if needed in future)
			raise NotImplementedError(f"Tubes with angle {tube_angle} are not supported yet.")

		return in_x_range and in_y_range

	# def get_agent_phase(self, agent: Agent, world: World):
	# 	"""Helper function to determine agent's current phase"""
	# 	pos = agent.state.p_pos
	# 	in_tube = self.is_in_tube(world, pos)
	# 	passed_tube = pos[1] < world.tube_params[self.current_tube]['exit'][1]
		
	# 	if not in_tube and not passed_tube:
	# 		return 0
	# 	elif in_tube:
	# 		return 1
	# 	else:
	# 		return 2

	def get_agent_phase(self, agent: Agent, world: World):
		"""Helper function to determine agent's current phase (supports horizontal and vertical tubes)"""
		
		pos = agent.state.p_pos
		tube = world.tube_params[self.current_tube[agent.id]]
		in_tube = self.is_in_tube(world, pos, agent)
		
		tube_angle = tube['angle']

		# If tube is vertical, check Y-coordinate for "passed_tube"
		if np.isclose(abs(tube_angle), np.pi / 2, atol=1e-2):
			passed_tube = pos[1] < tube['exit'][1]

		# If tube is horizontal, check X-coordinate for "passed_tube"
		elif np.isclose(tube_angle, 0, atol=1e-2) or np.isclose(abs(tube_angle), np.pi, atol=1e-2):
			if self.current_tube[agent.id] %2== 0:  # Agent ID is even
				passed_tube = pos[0] < tube['exit'][0]
				# print("Agent",agent.id,"passed_tube",passed_tube)
			else:
				passed_tube = pos[0] > tube['exit'][0]
				# print("Agent",agent.id,"passed_tube",passed_tube)

		else:
			raise NotImplementedError(f"Tubes with angle {tube_angle} are not supported yet.")

		if not in_tube and not passed_tube:
			return 0  # Before entering tube
		elif in_tube:
			return 1  # Inside tube
		else:
			return 2  # After exiting tube

	def get_agent_nearby_goals(self, agent: Agent, world: World):
		"""Helper function to get agent's nearby goals"""
		# print("Agent",agent.id,"obs")
		# === Find Two Closest Goals ===
		goal_dists = np.array([np.linalg.norm(agent.state.p_pos - l) for l in self.landmark_poses])
		sorted_goal_indices = np.argsort(goal_dists)  # Sort goal indices by distance
		top_two_indices = sorted_goal_indices[:2]  # Get the two closest goal indices
		min_dist = np.min(goal_dists)
		chosen_goal = top_two_indices[0]
		if min_dist < self.coordination_range:
			# print("\nwithin obs range of goal", min_dist)
			agents_goal = self.landmark_poses[chosen_goal]
			# find which goals are within self.coordination_range and unoccupied
			nearby_goals = np.where(goal_dists < self.coordination_range)[0]
			# check if any of those have bee occupied and verify if true
			for goal in nearby_goals:
				# print("Agent",agent.id,"Nearby goal",goal,self.landmark_poses_occupied[goal])
				if self.landmark_poses_occupied[goal] == self.coordination_range:
					# print("Agent",agent.id,"Nearby goal",goal, "is occupied")
					goal_proximity = np.array([np.linalg.norm(self.landmark_poses[goal] - agent.state.p_pos)  for agent in world.agents])
					if not np.any(goal_proximity < self.min_dist_thresh):
						# print("Agent",agent.id,"Nearby goal",goal, "is unoccupied")
						self.landmark_poses_occupied[goal] = 1 - np.min(goal_proximity)  # Reset falsely occupied goals

			if min_dist < self.min_dist_thresh:
				if agent.status == True:
					self.landmark_poses_occupied[top_two_indices[0]] = self.coordination_range
					# print("Agent",agent.id," AT GOAL",np.min(world.dists), "goal_occupied",self.landmark_poses_occupied[chosen_goal])
				else:
					self.landmark_poses_occupied[top_two_indices[0]] = self.coordination_range-min_dist
				self.goal_history[top_two_indices[0]] = agent.id
				# print("Ag",agent.id," near GOAL",np.min(world.dists), "goal_occupied",self.landmark_poses_occupied[chosen_goal])

			else:
				# goal_proximity is finding how many agents are near this chosen goal
				goal_proximity = np.array([np.linalg.norm(agents_goal - agent.state.p_pos)  for agent in world.agents])
				# print("Agent",agent.id,"chosen_goal", chosen_goal, "goal_proximity",goal_proximity, "flags",self.landmark_poses_occupied, "history",self.goal_history)
				closest_dist_to_goal = np.min(goal_proximity)


				# agent veered off the goal
				if self.landmark_poses_occupied[chosen_goal] == self.coordination_range:

					# if there are no agents on the goal, then the agent can take the goal and change the occupancy value
					if np.any(goal_proximity < self.min_dist_thresh):
						# print("Agent!", "{:.0f}".format(self.goal_history[chosen_goal]), " is already at goal", "{:.0f}".format(chosen_goal), "min_dist", "{:.3f}".format(min_dist), "occupied flags",  self.landmark_poses_occupied, "history", self.goal_history)

						######
						## Add case when all nearby observed goals are occupied
						unoccupied_goals = self.landmark_poses[self.landmark_poses_occupied!= self.coordination_range]
						unoccupied_goals_indices = np.where(self.landmark_poses_occupied != self.coordination_range)[0]
						# print ("Agent",agent.id,"unoccupied_goals",unoccupied_goals, "unoccupied_goals_indices",unoccupied_goals_indices)
						assert len(unoccupied_goals) > 0, f"All goals are occupied {self.landmark_poses_occupied}, {self.goal_history},{world.dists} {goal_proximity}"
						# input("Press Enter to continue...")
						chosen_goal = np.argmin(np.linalg.norm(agent.state.p_pos - unoccupied_goals, axis=1))
						agents_goal = unoccupied_goals[chosen_goal]

					else:
						## add assertion to see if no goal is falsely occupied
						assert not np.any(goal_proximity < self.min_dist_thresh), f"Agent {agent.id} is not at goal {chosen_goal} but flag is set to occupied"
						self.landmark_poses_occupied[chosen_goal] = self.coordination_range-closest_dist_to_goal

				# another agent already at goal, can't overwrite the flag
				elif self.landmark_poses_occupied[chosen_goal] != self.coordination_range:
					# print("No agent at this goal")
					self.landmark_poses_occupied[chosen_goal] = self.coordination_range-closest_dist_to_goal

			goal_occupied = np.array([self.landmark_poses_occupied[chosen_goal]])
			goal_history = self.goal_history[chosen_goal]

		else:
			# create another variable to store which goals are uncoccupied using an index of 0, 1 or 2 based on self.landmark_poses
			unoccupied_goals = self.landmark_poses[self.landmark_poses_occupied!= self.coordination_range]
			# print("unoccupied_goals",unoccupied_goals)
			unoccupied_goals_indices = np.where(self.landmark_poses_occupied != self.coordination_range)[0]
			if len(unoccupied_goals) > 0:

				## determine which goal from self.landmark_poses is this chosen unocccupied goal
				## use the index of the unoccupied goal to get the goal from self.landmark_poses
				min_dist_goal = np.argmin(np.linalg.norm(agent.state.p_pos - unoccupied_goals, axis=1))
				agents_goal = unoccupied_goals[min_dist_goal]
				# print("closest unoccupied goal",agents_goal)
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
		# input("Press Enter to continue...")
		return goal_pos, closest_goal_occupied, rel_second_closest_goal, goal_history


	def observation(self, agent: Agent, world: World) -> arr:
		"""
		Returns an observation for the agent, including:
		- Agent velocity
		- Agent position
		- Relative positions of the two nearest neighbors
		- Distances and occupancy status of the two closest goals
		- Tube-related information (distance to entrance/exit, width, and phase)
		"""
		agent_pos = agent.state.p_pos
		agent_heading = agent.state.theta
		agent_speed = agent.state.speed
		agent_vel = agent.state.p_vel
		
		## only single goal exists
		agents_goal = self.landmark_poses[self.goal_match_index[self.current_tube[agent.id],agent.id]]
		# print("agent id",agent.id, "current tube", self.current_tube[agent.id], "goal", agents_goal)
		goal_pos = agents_goal - agent.state.p_pos

		## TO DO all agents go to the same goal
		closest_goal_occupied =np.array([self.landmark_poses_occupied[self.goal_match_index[self.current_tube[agent.id],agent.id]]])

		rel_second_closest_goal = goal_pos
		# goal_pos, closest_goal_occupied, rel_second_closest_goal, goal_history = self.get_agent_nearby_goals(agent, world)
		# goal_pos = get_rotated_position_from_relative(goal_pos, agent_heading)
		# rel_second_closest_goal = get_rotated_position_from_relative(rel_second_closest_goal, agent_heading)
		# goal_heading = agents_goal.heading
		# goal_speed = agents_goal.speed


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


		# # Rotate nearest neighbors relative to agent's heading
		# rotated_neighbors = [
		# 	get_rotated_position_from_relative(neighbor, agent_heading)
		# 	for neighbor in nearest_neighbors
		# ]

		# # Flatten into a single array
		# rotated_neighbors = np.concatenate(rotated_neighbors)

		# Tube parameters
		tube_entrance = world.tube_params[self.current_tube[agent.id]]['entrance']
		tube_exit = world.tube_params[self.current_tube[agent.id]]['exit']
		tube_width = world.tube_params[self.current_tube[agent.id]]['width']
		# print("Agent",agent.id,"tubbe_entrance",tube_entrance)
		# print("tube_exit",tube_exit)
		# Calculate distances and directions to tube entrance/exit
		rel_to_entrance = tube_entrance - agent_pos
		rel_to_exit = tube_exit - agent_pos

		# rot_rel_entrance = get_rotated_position_from_relative(rel_to_entrance, agent_heading)
		# rot_rel_exit = get_rotated_position_from_relative(rel_to_exit, agent_heading)

		dist_to_entrance = np.linalg.norm(rel_to_entrance)
		dist_to_exit = np.linalg.norm(rel_to_exit)
		
		# Calculate phase
		in_tube = self.is_in_tube(world, agent_pos, agent)
		passed_tube = agent_pos[1] < tube_exit[1]  # Assuming tube exit is at bottom
		
		# if not in_tube and not passed_tube:
		# 	phase = 0  # Pre-tube phase
		# elif in_tube:
		# 	phase = 1  # In-tube phase
		# else:
		# 	if self.phase_reached[agent.id] == 0:
		# 		phase = 0
		# 	else:
		# 		phase = 2  # Post-tube phase
		phase = self.get_agent_phase(agent, world)
		tube_params = np.concatenate([
			rel_to_entrance,  # Vector to tube entrance
			rel_to_exit,     # Vector to tube exit
			[tube_width],    # Tube width
			[phase]          # Current phase
		])
		# print("self.current_tube",self.current_tube[agent.id])
		# print("tube_params",tube_params)

		return np.concatenate([
			agent_pos,
			agent_vel,
			goal_pos,
			closest_goal_occupied,
			rel_second_closest_goal,
			nearest_neighbors,
			tube_params
		])
	

	def get_id(self, agent:Agent) -> arr:
		return np.array([agent.global_id])


	
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

		disconnected_mask = []
		# for agent entity, disconnect if it is done or not departed
		for entity in world.agents:
			disconnected = entity.status
			# if entity.status:
				# print("agent_done",entity.id,disconnected)
			disconnected_mask.append(disconnected)

		# for landmark agent, disconnect if it is reached by the agent.
		for (i_landmark, landmark) in enumerate(world.landmarks):
			# landmark_agent_id = i_landmark % self.num_agents
			# landmark_order = i_landmark // self.num_agents
			# landmark_done = self.reached_goal[landmark_agent_id] > landmark_order
			## use goal_tracker to remove landmarks that are reached
			landmark_done = np.any(self.goal_tracker == landmark.id)
			# if landmark_done:
				# print("landmark_done",landmark.id)
			disconnected_mask.append(landmark_done)
			# print("landmark_done",landmark.id)
		# print("disconnected_mask",disconnected_mask)
		adj[disconnected_mask, :] = 0   # Mask rows for done agents
		adj[:, disconnected_mask] = 0   # Mask columns for done agents
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
			goal_pos = world.get_entity('landmark', self.goal_match_index[self.current_tube[entity.id],entity.id]).state.p_pos
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
		agent_speed = agent.state.speed
		entity_pos = entity.state.p_pos
		entity_speed = entity.state.speed
		rel_pos = entity_pos - agent_pos
		# rel_pos = get_rotated_position_from_relative(rel_pos, agent.state.theta)
		# rel_speed = entity_speed - agent_speed
		rel_vel = entity.state.p_vel - agent.state.p_vel
		if 'agent' in entity.name:
			# world.dists = np.array([np.linalg.norm(entity.state.p_pos - l) for l in self.landmark_poses])
			# min_dist = np.min(world.dists)
			# if min_dist < self.coordination_range:
			# 	# If the minimum distance is already less than self.min_dist_thresh, use the previous goal.
			# 	chosen_goal = np.argmin(world.dists)
			# 	goal_pos = self.landmark_poses[chosen_goal]
			# 	goal_history = self.goal_history[chosen_goal]
			# 	goal_occupied = np.array([self.landmark_poses_occupied[chosen_goal]])

			# else:
			# 	unoccupied_goals = self.landmark_poses[self.landmark_poses_occupied!= self.coordination_range]
			# 	unoccupied_goals_indices = np.where(self.landmark_poses_occupied != self.coordination_range)[0]
			# 	if len(unoccupied_goals) > 0:

			# 		## use closest goal

			# 		## determine which goal from self.landmark_poses is this chosen unocccupied goal
			# 		## use the index of the unoccupied goal to get the goal from self.landmark_poses
			# 		min_dist_goal = np.argmin(np.linalg.norm(entity.state.p_pos - unoccupied_goals, axis=1))
			# 		goal_pos = unoccupied_goals[min_dist_goal]
			# 		## check if the goal is occupied
			# 		goal_occupied = np.array([self.landmark_poses_occupied[unoccupied_goals_indices[min_dist_goal]]])
			# 		goal_history = self.goal_history[unoccupied_goals_indices[min_dist_goal]]

			# 	else:
			# 		# Handle the case when all goals are occupied.
			# 		goal_pos = entity.state.p_pos
			# 		self.landmark_poses_occupied = np.zeros(self.num_agents)
			# 		goal_occupied = np.array([self.landmark_poses_occupied[entity.id]])
			# 		goal_history = self.goal_history[entity.id]

			# goal_history = np.array([goal_history])
			goal_pos = self.landmark_poses[entity.id]
			rel_goal_pos = goal_pos - agent_pos
			goal_occupied = np.array(self.landmark_poses_occupied[self.goal_match_index[self.current_tube[entity.id],entity.id]])
			# rel_goal_pos = get_rotated_position_from_relative(rel_goal_pos, agent.state.theta)
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
		return np.hstack([rel_vel, rel_pos, rel_goal_pos,goal_occupied,entity_type])



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
