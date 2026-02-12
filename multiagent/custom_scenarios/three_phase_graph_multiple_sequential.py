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

# import scipy.spatial.distance as dist
import os,sys
sys.path.append(os.path.abspath(os.getcwd()))

# import numba
# from numba import cuda, jit

from multiagent.core import EntityDynamicsType, World, Agent, Landmark, Entity, Wall
from multiagent.scenario import BaseScenario
from multiagent.config import UnicycleVehicleConfig, DoubleIntegratorConfig, AirTaxiConfig
# from scipy.optimize import linear_sum_assignment
from multiagent.custom_scenarios.utils import *
# from marl_fair_assign import solve_fair_assignment

entity_mapping = {'agent': 0, 'landmark': 1, 'obstacle':2, 'wall':3}


# @staticmethod
# @jit(nopython=True)
def get_rotated_position_from_relative(relative_position: np.ndarray,
		reference_heading: float) -> np.ndarray:
	# returns relative position from the reference state.
	assert relative_position.shape == (2,), "relative_position should be a 2D array."
	rot_matrix = np.array([[np.cos(reference_heading), np.sin(reference_heading)], [-np.sin(reference_heading), np.cos(reference_heading)]])
	relative_position_rotated = np.dot(rot_matrix, relative_position)
	return relative_position_rotated


class Scenario(BaseScenario):

	def get_aspect_ratio_for_scenario(self) -> float:
		return 1.0

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
		self.world_aspect_ratio = self.get_aspect_ratio_for_scenario()
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

		self.current_tube = np.zeros(self.num_agents, dtype=int)
		self.max_tubes = 5
		self.tube_choice = 0

		# create heatmap matrix to determine the goal agent pairs
		self.goal_reached = np.full(self.num_agents, -1)
		self.wrong_goal_reached = np.zeros(self.num_agents)
		self.goal_matched = np.zeros(self.num_agents)

		self.goal_tracker = np.full(self.num_agents, -1)## 	keeps track of which goal each agent goes to using self.goal_tracker[agent.id] = self.goal_match_index[agent.id]

		self.conformance_percent = np.zeros(self.num_agents)

		self.delta_spacing = []

		self.spacing_violation = np.zeros(self.num_agents)
		if args.dynamics_type == 'unicycle_vehicle':
			self.dynamics_type = EntityDynamicsType.UnicycleVehicleXY
			self.config_class = UnicycleVehicleConfig
			self.min_turn_radius = 0.5 * (UnicycleVehicleConfig.V_MAX + UnicycleVehicleConfig.V_MIN) / UnicycleVehicleConfig.ANGULAR_RATE_MAX

		elif args.dynamics_type == 'double_integrator':
			self.dynamics_type = EntityDynamicsType.DoubleIntegratorXY
			self.config_class = DoubleIntegratorConfig
			self.min_turn_radius = 0.0
		
		elif args.dynamics_type == 'air_taxi':
			self.dynamics_type = EntityDynamicsType.AirTaxiXY
			self.config_class = AirTaxiConfig
			self.min_turn_radius = 0.0
		else:
			raise NotImplementedError
		self.coordination_range = self.config_class.COORDINATION_RANGE
		self.min_dist_thresh = self.config_class.DISTANCE_TO_GOAL_THRESHOLD
		self.separation_distance = self.config_class.COLLISION_DISTANCE

		self.phase_reached = np.zeros(self.num_agents)  ## keeps track of which phase each agent is in when it first enters it

		self.phase_reward_cooldown_steps = self.episode_length                   # steps to cool down 0->1 reward

		# scripted agent dynamics are fixed to double integrator for now (as it was originally)
		scripted_agent_dynamics_type = EntityDynamicsType.DoubleIntegratorXY

		# needed when scenario is used in evaluation.
		self.curriculum_ratio = 1.0
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
		world.world_aspect_ratio = self.world_aspect_ratio

		if hasattr(self, 'with_background'):
			world.with_background = self.with_background
		else:
			world.with_background = False
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

		self.goal_match_index = np.arange(self.num_agents)
		self.goal_history = np.full(self.num_agents, -1)
		self.goal_reached =  np.full(self.num_agents, -1)

		self.goal_tracker = np.full(self.num_agents, -1)
		self.conformance_percent = np.zeros(self.num_agents)
		self.delta_spacing =[]
		self.spacing_violation = np.zeros(self.num_agents)
		self.steps_in_corridor = np.zeros(self.num_agents)

		self.current_tube = np.zeros(self.num_agents, dtype=int)
		self.tube_choice = 0

		self.agent_dist_traveled = np.zeros(self.num_agents)
		self.agent_time_taken = np.zeros(self.num_agents)
		wall_length = np.random.uniform(0.2, 0.8)
		self.wall_length = wall_length * self.world_size/4

		self.phase_reached = np.zeros(self.num_agents)
		self.entry_reward_cooldown = np.zeros(self.num_agents, dtype=np.int32)
		# Store previous longitudinal position (s) for progress reward
		self.prev_proj = np.zeros(self.num_agents, dtype=np.float32)
		self.prev_goal_dist = np.full(self.num_agents, np.inf, dtype=np.float32)

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
		# Initialize progress baselines after positions/goals are set
		for agent in world.agents:
			s, _, _, _ = self._tube_coords(world, agent.state.p_pos, self.current_tube[agent.id])
			self.prev_proj[agent.id] = s
			goal_pos = self.landmark_poses[self.goal_match_index[agent.id]]
			self.prev_goal_dist[agent.id] = np.linalg.norm(agent.state.p_pos - goal_pos)
	
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
		# Initialize tube parameters
		self.setup_s_tube_params(world)

		num_agents_added = 0
		agents_added = []
		boundary_thresh = 0.99
		current_tube = world.tube_params[0]

		# Staggered queue behind entrance — agents line up along the
		# corridor axis with guaranteed minimum longitudinal spacing.
		min_sep = max(3.0 * self.separation_distance, 1.5*self.separation_distance)
		long_spacing = min_sep * 1.2
		lateral_spread = self.world_size * 0.3

		while True:
			if num_agents_added == self.num_agents:
				break

			# All agents start on tube 0 (sequential traversal)
			self.current_tube[num_agents_added] = 0

			# Place agents behind their corridor entrance using corridor frame
			corridor_e = current_tube['e']  # unit vec: entrance → exit
			corridor_n = current_tube['n']  # unit vec: left-hand normal
			entrance = current_tube['entrance']

			# Stagger behind entrance along corridor axis
			along_offset = -(num_agents_added + 1) * long_spacing
			lateral_jitter = np.random.uniform(-1.0, 1.0) * lateral_spread
			along_jitter = np.random.uniform(-0.2, 0.2) * long_spacing

			random_pos = (entrance
						  + (along_offset + along_jitter) * corridor_e
						  + lateral_jitter * corridor_n)

			agent_size = world.agents[num_agents_added].size
			obs_collision = self.is_obstacle_collision(random_pos, agent_size, world)
			if obs_collision:
				lateral_jitter = np.random.uniform(-1.0, 1.0) * lateral_spread * 0.5
				random_pos = (entrance
							  + (along_offset + along_jitter) * corridor_e
							  + lateral_jitter * corridor_n)

			world.agents[num_agents_added].state.p_pos = random_pos
			# Initialize heading toward the corridor entrance
			corridor_heading = np.arctan2(corridor_e[1], corridor_e[0])
			init_heading = corridor_heading + np.random.uniform(-np.pi/6, np.pi/6)
			world.agents[num_agents_added].state.reset_velocity(theta=init_heading)
			world.agents[num_agents_added].state.c = np.zeros(world.dim_c)
			world.agents[num_agents_added].status = False
			agents_added.append(world.agents[num_agents_added])
			num_agents_added += 1
		# agent_pos = [agent.state.p_pos for agent in world.agents]
		#####################################################

		self.agent_id_updated = np.arange(self.num_agents)
		# Place each agent's landmark at the exit of its own assigned tube
		for agent in world.agents:
			agent_tube = world.tube_params[self.current_tube[agent.id]]
			landmark_pos = self._goal_pos_for_tube(agent_tube)
			landmark_idx = self.goal_match_index[agent.id]
			world.landmarks[landmark_idx].state.p_pos = landmark_pos
			world.landmarks[landmark_idx].state.reset_velocity()

		# Update landmark poses arrays
		self.landmark_poses = np.array([landmark.state.p_pos for landmark in world.landmarks])
		# print("landmark pose",self.landmark_poses)
		self.landmark_poses_occupied = np.zeros(self.num_agents)
		self.landmark_poses_updated = np.array([landmark.state.p_pos for landmark in world.landmarks])
		self.agent_id_updated = np.arange(self.num_agents)
		#####################################################

		############ find minimum times to goals ############
		if self.max_speed is not None:
			for agent in world.agents:
				self.min_time(agent, world)
		#####################################################

	# setup S-shaped corridor configuration using _build_tube_params
	def setup_s_tube_params(self, world):
		"""
		Set up 5 corridors in an S-shaped configuration.
		Agents enter from the left at the top and traverse:
		  Tube 0: Horizontal left→right  (top of S)
		  Tube 1: Vertical going down    (right side)
		  Tube 2: Horizontal right→left  (middle of S)
		  Tube 3: Vertical going down    (left side)
		  Tube 4: Horizontal left→right  (bottom of S)
		Each tube's exit connects to the next tube's entrance.
		"""
		world.tube_params = []

		# tube_width = max(
		# 	3 * world.agents[0].size * 2.5,
		# 	self.world_size * 0.15
		# )
		tube_width = 0.8
		ws = self.world_size

		# Horizontal arm length and vertical connector length
		h_len = ws * 0.5
		v_len = ws * 0.4           # longer vertical connectors (tubes 1 & 3)

		# Gap between consecutive corridors (transition space for agents to turn)
		gap = ws * 0.0             # wider gap at each corner

		# Y-positions of the three horizontal arms (spread further apart to fit gaps)
		y_top = v_len + gap
		y_mid = 0.0
		y_bot = -(v_len + gap)

		# X-extents
		x_left = -h_len / 2
		x_right = h_len / 2

		# Inset: vertical tubes sit slightly inside the horizontal endpoints
		# so entrances don't align exactly with the previous tube's exit.
		inset = ws * -0.08

		# Define the 5 tubes with gaps + inset at each corner.
		tube_defs = [
			(np.array([x_left,            y_top]),       np.array([x_right,           y_top])),        # Tube 0: top L→R
			(np.array([x_right - inset,   y_top - gap]), np.array([x_right - inset,   y_mid + gap])),  # Tube 1: right ↓ (inset)
			(np.array([x_right,           y_mid]),       np.array([x_left,            y_mid])),        # Tube 2: mid R→L
			(np.array([x_left + inset,    y_mid - gap]), np.array([x_left + inset,    y_bot + gap])),  # Tube 3: left ↓ (inset)
			(np.array([x_left,            y_bot]),       np.array([x_right,           y_bot])),        # Tube 4: bot L→R
		]

		total_length = 0.0
		for entrance, exit_pt in tube_defs:
			length = float(np.linalg.norm(exit_pt - entrance))
			corridor_vec = (exit_pt - entrance) / length
			# Compute angle consistent with _build_tube_params convention
			angle = float(np.arctan2(-corridor_vec[0], -corridor_vec[1]))
			rotation_matrix = np.array([
				[np.cos(angle), np.sin(angle)],
				[-np.sin(angle), np.cos(angle)]
			])
			tube = self._build_tube_params(
				entrance, exit_pt, tube_width, angle, length, rotation_matrix
			)
			world.tube_params.append(tube)
			total_length += length

		avg_length = total_length / len(tube_defs)
		self.progress_gain = self.goal_rew / (avg_length * 10)


	def _goal_pos_for_tube(self, tube):
		"""Compute the landmark/goal position for a given tube (placed past its exit along corridor axis)."""
		# Place goal just past exit along the corridor direction
		offset = self.world_size / 20
		return np.array(tube['exit']) + offset * np.array(tube['e'])

	def _build_tube_params(self, entrance, exit, width, angle, length, rotation_matrix):
		"""
		Helper function to build tube parameters dictionary.
		Reusable for creating multiple tubes with consistent structure.
		"""
		# Calculate perpendicular direction for formation lines
		perpendicular_angle = angle + np.pi/2
		formation_direction = np.array([
			np.cos(perpendicular_angle),
			np.sin(perpendicular_angle)
		])
		
		# Calculate line formation parameters
		line_length = width * 0.8
		half_line = (line_length/2) * formation_direction
		
		# Pre-tube formation line (before entrance)
		pre_tube_center = entrance + rotation_matrix @ np.array([0, self.world_size * 0.15])
		pre_tube_line = {
			'start_pos': pre_tube_center - half_line,
			'end_pos': pre_tube_center + half_line,
			'angle': perpendicular_angle
		}
		
		# Post-tube target line (after exit)
		post_tube_center = exit - rotation_matrix @ np.array([0, self.world_size * 0.15])
		post_tube_line = {
			'start_pos': post_tube_center - half_line,
			'end_pos': post_tube_center + half_line,
			'angle': perpendicular_angle
		}
		
		# Precompute tube frame for fast queries
		L = float(np.linalg.norm(exit - entrance)) + 1e-9
		corridor_vec = (exit - entrance) / L
		n_vec = np.array([-corridor_vec[1], corridor_vec[0]], dtype=np.float32)
		
		# Full-width entrance gate settings (tunable)
		self.gate_front_ratio = getattr(self, 'gate_front_ratio', 0.08)  # inside tube
		self.gate_back_ratio = getattr(self, 'gate_back_ratio', 0.02)    # just outside entrance
		
		# Full-width exit gate settings (tunable)
		self.exit_back_ratio = getattr(self, 'exit_back_ratio', 0.02)    # inside tube near exit
		self.exit_front_ratio = getattr(self, 'exit_front_ratio', 0.08)  # just outside exit	
		
		return {
			'entrance': entrance,
			'exit': exit,
			'width': width,
			'angle': angle,
			'length': length,
			'pre_tube_line': pre_tube_line,
			'post_tube_line': post_tube_line,
			'rotation_matrix': rotation_matrix,
			'formation_direction': formation_direction,
			'e': corridor_vec,
			'n': n_vec,
			'L': L,
			'half_width': float(width) * 0.5,
			'gate_front_ratio': self.gate_front_ratio,
			'gate_back_ratio': self.gate_back_ratio,
			'exit_back_ratio': self.exit_back_ratio,
			'exit_front_ratio': self.exit_front_ratio
		}


	def setup_tube_params(self, world):
		"""
		Set up tube parameters using modified landmark line logic
		"""
		# # Initialize tube list
		# world.tube_params = []
		# Calculate tube width based on number of agents
		self.tube_width = max(
			3 * world.agents[0].size * 2.5,  # Width based on agents # =3  TODO: harcoded
			self.world_size * 0.15  # Minimum width
		)

		# random_angle = np.random.uniform(-np.pi/2, np.pi/2)
		random_angle = 0.0
		# print(f"Random Angle: {random_angle*180/np.pi} degrees")
		# Calculate tube length
		tube_length = self.world_size * 0.8  # Use 80% of world size for tube length
		tube_length += np.random.uniform(-self.world_size*0.3, self.world_size*0.1)  # Add some randomness
		
		# Scales: make 1 full tube traversal worth ~goal_rew
		self.progress_gain = self.goal_rew / (tube_length*10)
		# Calculate center point of the world
		world_center = np.array([0, 0])
		
		# Calculate entrance and exit points using rotation
		# Start with vertical positions
		base_entrance = np.array([0, tube_length/4])  # Start above center
		base_exit = np.array([0, -tube_length/4])     # End below center
		
		# Create rotation matrix
		rotation_matrix = np.array([
			[np.cos(random_angle), np.sin(random_angle)],
			[-np.sin(random_angle), np.cos(random_angle)]
		])
		
		# Apply rotation to entrance and exit points
		entrance = world_center + rotation_matrix @ base_entrance
		# print("rotation_matrix @ base_entrance", rotation_matrix @ base_entrance)
		exit = world_center + rotation_matrix @ base_exit
		# Store tube parameters
		world.tube_params = {
			'entrance': entrance,
			'exit': exit,
			'width': self.tube_width,
			'angle': random_angle,
			'length': tube_length
		}

		# print(f"Tube Entrance: {entrance}, Exit: {exit}, Angle: {random_angle*180/np.pi} degrees")
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

		# Precompute tube frame for fast queries
		L = float(np.linalg.norm(exit - entrance)) + 1e-9
		corridor_vec = (exit - entrance) / L
		n_vec = np.array([-corridor_vec[1], corridor_vec[0]], dtype=np.float32)  # left-hand normal
		world.tube_params.update({
			'e': corridor_vec,
			'n': n_vec,
			'L': L,
			'half_width': float(self.tube_width) * 0.5
		})

		# Full-width entrance gate settings (tunable)
		self.gate_front_ratio = getattr(self, 'gate_front_ratio', 0.08)  # inside tube
		self.gate_back_ratio = getattr(self, 'gate_back_ratio', 0.02)    # just outside entrance
		
		# Full-width exit gate settings (tunable)
		self.exit_back_ratio = getattr(self, 'exit_back_ratio', 0.02)    # inside tube near exit
		self.exit_front_ratio = getattr(self, 'exit_front_ratio', 0.08)  # just outside exit	

	# --- Shared geometry helpers (reduce redundancy) ---
	def _tube_frame(self, world: World, current_tube: int = 0):

		tp = world.tube_params[current_tube]
		return tp['entrance'], tp['e'], tp['n'], float(tp['L']), float(tp['half_width'])

	def _tube_coords(self, world: World, pos: np.ndarray, current_tube: int = 0):
		"""Return (s, y, L, half_w): longitudinal s from entrance and signed lateral y."""
		entrance, e, n, L, half_w = self._tube_frame(world, current_tube)
		r = np.asarray(pos, dtype=np.float32) - entrance  # r: 2D vector from the tube entrance to the queried position, in world coordinates. r = pos − entrance.
		s = float(np.dot(r, e))  # s is the along-tube coordinate from the entrance plane (s=0 at the entrance, s>0 inside the tube, s<0 before the entrance).
		y = float(np.dot(r, n))  # y is the signed lateral offset from the tube centerline (y=0 on centerline, |y| increases outward).
		return s, y, L, half_w

	def _in_tube_rect(self, s: float, y: float, L: float, half_w: float, eps: float = 0.05) -> bool:
		return (-eps <= s <= L + eps) and (abs(y) <= half_w + eps)
	
	def _in_entrance_gate(self, s: float, y: float, L: float, half_w: float, eps: float = 0.05) -> bool:
		"""Full-width gate spanning the entrance edge: s in [-gate_back, +gate_front], |y|<=half_w."""
		gate_front = float(self.gate_front_ratio) * L
		gate_back = float(self.gate_back_ratio) * L
		return (-gate_back - eps <= s <= gate_front + eps) and (abs(y) <= half_w + eps)
	
	# Distance to the full-width entrance edge (rectangle at s=0 with |y|<=half_w)
	def _entrance_gate_distance(self, s: float, y: float, half_w: float) -> float:
		# nearest point on entrance edge: (s'=0, y' clamped to [-half_w, half_w])
		clamped_y = float(np.clip(y, -half_w, half_w))
		ds = abs(float(s))   # before entrance plane → distance along axis
		dy = float(y) - clamped_y       # lateral overflow outside corridor width
		return float(np.hypot(ds, dy))
	
	# Full-width exit gate: s in [L - exit_back, L + exit_front], |y| <= half_w
	def _in_exit_gate(self, s: float, y: float, L: float, half_w: float, eps: float = 0.05) -> bool:
		exit_back = float(self.exit_back_ratio) * L
		exit_front = float(self.exit_front_ratio) * L
		return (L - exit_back - eps <= s <= L + exit_front + eps) and (abs(y) <= half_w + eps)

	# Distance to the exit edge segment (plane s=L, clamped laterally)
	def _exit_gate_distance(self, s: float, y: float, L: float, half_w: float, penalize_backward: bool = False) -> float:
		"""
		If penalize_backward is False, only measure remaining distance inside the tube (L - s)+.
		If True, use |s - L| (useful if already beyond exit but you want symmetric distance).
		"""
		clamped_y = float(np.clip(y, -half_w, half_w))
		ds = max(0.0, float(L - s)) if not penalize_backward else abs(float(s - L))
		dy = float(y) - clamped_y
		return float(np.hypot(ds, dy))

	# --- Optimized queries using the frame ---
	def is_in_tube(self, world: World, pos: np.ndarray, current_tube: int = 0) -> bool:
		s, y, L, half_w = self._tube_coords(world, pos, current_tube=current_tube)
		return self._in_tube_rect(s, y, L, half_w)
	
	def get_agent_phase(self, agent: Agent, world: World) -> int:
		pos = agent.state.p_pos
		current_tube = self.current_tube[agent.id]
		# in_tube = self.is_in_tube(world, pos)
		s, y, L, half_w = self._tube_coords(world, pos, current_tube=current_tube)
		in_tube = self._in_tube_rect(s, y, L, half_w)

		passed_tube = (s > L)
		valid_entrance = self._in_entrance_gate(s, y, L, half_w)
		valid_exit = self._in_exit_gate(s, y, L, half_w)
		# if agent.id == 2:
		# 	print("Agent", agent.id, "position", pos, "in_tube:", in_tube)
		# 	print(f"valid exit: {valid_exit}, passed_tube: {passed_tube}, s: {float(s):.3f}, y: {float(y):.3f}, L: {float(L):.3f}, half_w: {float(half_w):.3f}")
		# Decrement cooldown here once per call
		if self.entry_reward_cooldown[agent.id] > 0:
			# print("Decrementing cooldown for agent", agent.id, "from", self.entry_reward_cooldown[agent.id])
			self.entry_reward_cooldown[agent.id] -= 1

		# Only check valid_entrance when transitioning from phase 0 to 1
		if not in_tube and not passed_tube:
			if not hasattr(agent, 'previous_phase'):
				# print("Agent", agent.id, "is in pre-tube phase")
				agent.previous_phase = 0
			# print("Agent {} is in pre-tube phase 000".format(agent.id))
			return 0  # Pre-tube phase
		elif in_tube:
			if not hasattr(agent, 'previous_phase'):
				# print("Agent", agent.id, "is in pre-tube phase")
				agent.previous_phase = 0
			if agent.previous_phase == 0:
				if valid_entrance:
					# print("Agent", agent.id, "entered tube correctly 1111")
					# agent.previous_phase = 1
					return 1  # Entered correctly
				else:
					return 0  # Reset if entered incorrectly
			else:
				# print("Agent", agent.id, "is in tube phase 1111")
				# agent.previous_phase = 1

				if agent.previous_phase == 1 and valid_exit:
					# if passed_tube and valid_exit:
					# print("Agent", agent.id, "exited tube correctly YOOO")
						# agent.previous_phase = 2
					# agent.previous_phase = 2
					return 2
				if agent.previous_phase == 2 and valid_exit:
					# print("Agent", agent.id, "---- still in tube but in the exit gate YOOO")
					return 2
				# print("Agent", agent.id, "is in tube phase 1111", valid_exit)
				return 1  # Already in tube, stay in phase 1
		if passed_tube:
			if self.phase_reached[agent.id] >= 1:
				if agent.previous_phase == 1 and valid_exit:
					# if passed_tube and valid_exit:
					# print("Agent", agent.id, "exited tube correctly 2222")
						# agent.previous_phase = 2
					# agent.previous_phase = 2
					return 2
				elif agent.previous_phase == 2:
					# print("Agent", agent.id, "is in post-tube phase 2222")
					return 2
				# elif agent.previous_phase == 1 and self.phase_reached[agent.id] == 2:  # corner case of exit gate condition where agent can be in the gate but still considered in phase 1
				# 	print("Agent", agent.id, "is in post-tube phase 2222")
				# 	return 2
				# print("Agent", agent.id, "didn't correctly exit tube  0000")
				# agent.previous_phase = 0
				return 0  # Post-tube phase
		# Default: Phase 0
		# agent.previous_phase = 0
		return 0

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
			'Dist_to_goal': np.float32(world.dist_left_to_goal[agent.id]),  # make this into float using numpy float32
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
			'Phase_reached': self.phase_reached[agent.id],


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

	def reward(self, agent: Agent, world: World) -> float:
		rew = 0.0
		if agent.status:
			return rew
		current_phase = self.get_agent_phase(agent, world)

		# --- Collision penalties (matching trained scenario) ---
		if agent.collide:
			for a in world.agents:
				if a.id == agent.id:
					continue
				if a.status:
					continue
				dist_aa = np.linalg.norm(agent.state.p_pos - a.state.p_pos)
				# --- Hard collision penalty ---
				if dist_aa < self.separation_distance:
					rew -= self.collision_rew
				# --- Soft proximity penalty (only when closing in) ---
				warning_zone = 2.5 * self.separation_distance
				if dist_aa < warning_zone:
					rel_pos = a.state.p_pos - agent.state.p_pos
					rel_vel = np.asarray(a.state.p_vel) - np.asarray(agent.state.p_vel)
					closing = -float(np.dot(rel_pos, rel_vel)) / (dist_aa + 1e-9)
					if closing > 0:
						proximity_ratio = (warning_zone - dist_aa) / (warning_zone - self.separation_distance + 1e-9)
						proximity_ratio = np.clip(proximity_ratio, 0.0, 1.0)
						rew -= proximity_ratio * self.collision_rew * 0.3

		# Calculate tube length
		current_tube = world.tube_params[self.current_tube[agent.id]]
		tube_direction = current_tube['exit'] - current_tube['entrance']
		tube_length = np.linalg.norm(tube_direction)
		agent_pos = agent.state.p_pos
		agent_heading = agent.state.theta
		# Use corridor direction for front/back (not agent heading)
		corridor_e = current_tube['e']
		s, y, L, half_w = self._tube_coords(world, agent_pos, self.current_tube[agent.id])
		valid_exit = self._in_exit_gate(s, y, L, half_w)
		# compute progress
		delta_proj = s - self.prev_proj[agent.id]

		front_agents = []
		back_agents = []
		for other in world.agents:
			if other is agent:
				continue
			rel_vec = other.state.p_pos - agent_pos
			proj = np.dot(rel_vec, corridor_e)
			if proj > 0:
				front_agents.append((proj, other))
			else:
				back_agents.append((proj, other))

		# Get closest in front and back
		front_agent = min(front_agents, key=lambda x: x[0])[1] if front_agents else None
		back_agent = max(back_agents, key=lambda x: x[0])[1] if back_agents else None

		# Calculate desired spacing based on tube length and number of agents
		desired_spacing = self.separation_distance
		
		# Track agent's previous phase if not already stored
		if not hasattr(agent, 'previous_phase'):
			agent.previous_phase = 0
		
		# Modified phase transition handling
		if current_phase == 2 and current_phase > agent.previous_phase + 1:
			rew -= self.goal_rew
		tube_direction_vector = tube_direction / tube_length
		entrance_to_agent = agent.state.p_pos - current_tube['entrance']

		if current_phase == agent.previous_phase+1 and self.phase_reached[agent.id] == current_phase-1:
			if current_phase == 1 and self._in_entrance_gate(s, y, L, half_w) and (self.entry_reward_cooldown[agent.id] == 0):
				rew += self.goal_rew
				self.entry_reward_cooldown[agent.id] = self.phase_reward_cooldown_steps
				self.phase_reached[agent.id] = 1
			elif current_phase == 2:
				rew += self.goal_rew
				self.phase_reached[agent.id] = 2

		# Phase-specific rewards
		if current_phase == 0:  # Pre-tube phase
			s, y, L, half_w = self._tube_coords(world, agent.state.p_pos, self.current_tube[agent.id])
			dist_to_entrance_edge = self._entrance_gate_distance(s, y, half_w)
			# Normalized distance penalty
			rew -= (dist_to_entrance_edge / (L + 1e-9)) * self.goal_rew * 0.5
			# Reward forward progress along corridor axis
			rew += self.progress_gain * max(delta_proj, -0.05)
			self.prev_proj[agent.id] = s
			# Reward forward velocity along corridor direction
			forward_speed = float(np.dot(agent.state.p_vel, corridor_e))
			rew += self.progress_gain * max(forward_speed, 0.0)
			# Heading alignment
			corridor_heading = np.arctan2(corridor_e[1], corridor_e[0])
			heading_error = abs((agent_heading - corridor_heading + np.pi) % (2*np.pi) - np.pi)
			if dist_to_entrance_edge < self.world_size * 0.1:
				rew -= heading_error * self.formation_rew * 0.5
			# Lateral penalty
			if abs(y) > half_w:
				rew -= (abs(y) - half_w) / (L + 1e-9) * self.formation_rew * 0.3
		elif current_phase == 1:  # In-tube phase
			spacing_error = 0
			if front_agent:
				diff = np.linalg.norm(front_agent.state.p_pos - agent.state.p_pos) - desired_spacing
				spacing_error += np.abs(diff) if diff < 0 else 0
				# Speed-matching: only when Euclidean distance < 1.5× sep_dist
				euc_dist = np.linalg.norm(front_agent.state.p_pos - agent.state.p_pos)
				if euc_dist < 1.5 * self.separation_distance:
					speed_diff = abs(agent.state.speed - front_agent.state.speed)
					rew -= speed_diff * self.formation_rew * 0.3
			if back_agent:
				diff = np.linalg.norm(back_agent.state.p_pos - agent.state.p_pos) - desired_spacing
				spacing_error += np.abs(diff) if diff < 0 else 0
			if spacing_error > 0:
				self.spacing_violation[agent.id] += 1
			rew -= spacing_error * self.formation_rew
			dist_to_exit_edge = self._exit_gate_distance(s, y, L, half_w)
			# Normalize by tube length
			rew -= (dist_to_exit_edge / (L + 1e-9)) * self.goal_rew * 0.3
			self.steps_in_corridor[agent.id] += 1
			self.delta_spacing.append(spacing_error)
			# Reward forward progress through the tube
			rew += self.progress_gain * max(delta_proj, -0.05)
			self.prev_proj[agent.id] = s
			# Reward forward velocity along corridor direction
			forward_speed = float(np.dot(agent.state.p_vel, corridor_e))
			rew += self.progress_gain * max(forward_speed, 0.0)

			corridor_heading = np.arctan2(corridor_e[1], corridor_e[0])
			heading_error = abs((agent_heading - corridor_heading + np.pi) % (2*np.pi) - np.pi)
			rew -= heading_error * self.formation_rew * 0.1
		elif current_phase == 2 and self.current_tube[agent.id] < self.max_tubes - 1:
			# Exited tube N → seamlessly switch to tube N+1
			next_tube_idx = self.current_tube[agent.id] + 1
			self.current_tube[agent.id] = next_tube_idx
			self.phase_reached[agent.id] = 0
			agent.previous_phase = 0
			self.entry_reward_cooldown[agent.id] = 0
			# Re-initialize progress baseline for the new tube
			s2, _, _, _ = self._tube_coords(world, agent.state.p_pos, next_tube_idx)
			self.prev_proj[agent.id] = s2
			# Move only THIS agent's landmark to the next tube's exit
			landmark_idx = self.goal_match_index[agent.id]
			world.landmarks[landmark_idx].state.p_pos = self._goal_pos_for_tube(world.tube_params[next_tube_idx])
			world.landmarks[landmark_idx].state.reset_velocity()
			self.landmark_poses[landmark_idx] = world.landmarks[landmark_idx].state.p_pos
			# Give a small bonus for completing a corridor
			rew += self.goal_rew * 0.5
			# Reset prev_goal_dist so Phase 2 progress reward starts fresh
			self.prev_goal_dist[agent.id] = np.inf
			# Recompute tube coords for the NEW tube so downstream checks use correct values
			s, y, L, half_w = self._tube_coords(world, agent.state.p_pos, next_tube_idx)
			current_tube = world.tube_params[next_tube_idx]
			corridor_e = current_tube['e']
			valid_exit = self._in_exit_gate(s, y, L, half_w)
			current_phase = 0
		elif current_phase == 2 and self.current_tube[agent.id] == self.max_tubes - 1:  # Post-tube phase (final corridor)
			dist_to_goal = np.linalg.norm(agent.state.p_pos - self.landmark_poses[self.goal_match_index[agent.id]])
			if dist_to_goal < self.min_dist_thresh:
				if agent.status is False:
					agent.status = True
					agent.state.reset_velocity()
					rew += self.goal_rew*5
			else:
				# Normalized distance penalty
				rew -= dist_to_goal / (self.world_size * 0.5 + 1e-9) * self.goal_rew * 0.5

				corridor_heading = np.arctan2(corridor_e[1], corridor_e[0])
				heading_error = abs((agent_heading - corridor_heading + np.pi) % (2 * np.pi) - np.pi)
				heading_thresh = float(self.config_class.GOAL_HEADING_THRESHOLD)
				if heading_error > heading_thresh:
					rew -= (heading_error - heading_thresh) * self.formation_rew * 0.02

				# Progress reward toward goal
				if np.isfinite(self.prev_goal_dist[agent.id]):
					delta_goal = self.prev_goal_dist[agent.id] - dist_to_goal
					rew += self.progress_gain * 3.0 * max(delta_goal, -0.1)
				self.prev_goal_dist[agent.id] = dist_to_goal


		# print("Agent.status",agent.status)
		if self.phase_reached[agent.id] == 1 and current_phase == 0:
			# print("Agent",agent.id,"left corridor")
			self.conformance_percent[agent.id] += 1
			# print("conformance_percent",self.conformance_percent[agent.id])

		if current_phase > self.phase_reached[agent.id]:
			# print(f"Agent {agent.id} reached phase {current_phase}")
			self.phase_reached[agent.id] = current_phase  # Update max phase reached globally
		## penalize for moving from higher phase to lower phase
		if current_phase < agent.previous_phase:
			# print(f"Agent {agent.id} tried to move back to phase {current_phase} from {agent.previous_phase}")
			rew -= self.collision_rew  #*4
			# print(f"Agent {agent.id} tried to move back to phase {current_phase} from {agent.previous_phase} rew", rew)
		if current_phase < self.phase_reached[agent.id]:
			rew -= self.collision_rew
			# print(f"Agent {agent.id} tried to move back to phase {current_phase} from {self.phase_reached[agent.id]} rew", rew)
		# Store current phase for next step
		agent.previous_phase = current_phase
		if self._in_tube_rect(s, y, L, half_w) and not current_phase == 1 and not valid_exit:
			rew -= self.collision_rew  #*2
			# print(f"Agent {agent.id} is in tube but not in phase 1 rew", rew)

		# If agent is past exit plane but never entered corridor: continuous penalty
		if s > L and self.phase_reached[agent.id] < 1:
			rew -= self.goal_rew
			# print(f"Agent {agent.id} skipped corridor (s={s:.2f} > L={L:.2f}): penalty {self.goal_rew}")
		
		# if current_phase == 2:
		# print(f"Agent {agent.id} total reward: ", rew)
		# input("Reward calculation complete for agent {}".format(agent.id))
		# input("Press Enter to continue...")

		return np.clip(rew, -4*self.collision_rew, self.goal_rew*5)

	def observation(self, agent: Agent, world: World) -> arr:
		"""
		Returns an observation for the agent with rotation invariance:
		- Keep the same feature order/length as before to preserve training scripts.
		- All relative vectors (goal, neighbors, tube entrance/exit) are rotated into
		the agent's heading frame so that the agent's x-axis aligns with its heading.
		"""
		# --- Agent state ---
		agent_pos = agent.state.p_pos
		agent_heading = float(getattr(agent.state, "theta", getattr(agent.state, "p_ang", 0.0)))
		agent_speed = float(getattr(agent.state, "speed", np.linalg.norm(agent.state.p_vel)))
		agent_vel = np.asarray(agent.state.p_vel, dtype=np.float32)


		# Rotate own velocity into ego frame
		rot_agent_vel = get_rotated_position_from_relative(agent_vel, agent_heading).astype(np.float32)

		# --- Goal related (single fair goal) ---
		goal_world_pos = self.landmark_poses[self.goal_match_index[agent.id]]
		# print("Agent", agent.id, "goal_world_pos", goal_world_pos)
		rel_goal_vec_world = goal_world_pos - agent_pos
		goal_pos = get_rotated_position_from_relative(rel_goal_vec_world, agent_heading).astype(np.float32)
		# print("Rotated goal_pos", goal_pos)

		# --- Two nearest neighbors (rotated) + closing rate ---
		neighbor_dists = []
		for other in world.agents:
			if other is agent:
				continue
			# Skip completed agents (they're "ghosts")
			if other.status:
				continue
			rel_pos_world = other.state.p_pos - agent_pos
			rel_vel_world = np.asarray(other.state.p_vel, dtype=np.float32) - agent_vel
			dist = float(np.linalg.norm(rel_pos_world))
			neighbor_dists.append((dist, rel_pos_world, rel_vel_world))

		neighbor_dists.sort(key=lambda x: x[0])
		nearest = [n[1] for n in neighbor_dists[:2]]
		while len(nearest) < 2:
			nearest.append(np.zeros(world.dim_p, dtype=np.float32))

		# Closing rate to nearest neighbor (positive = approaching)
		if neighbor_dists:
			nn_dist, nn_rel_pos, nn_rel_vel = neighbor_dists[0]
			if nn_dist > 1e-6:
				closing_rate = -float(np.dot(nn_rel_pos, nn_rel_vel)) / nn_dist
			else:
				closing_rate = 0.0
			# Normalize by twice max speed
			closing_rate_norm = np.clip(closing_rate / (2.0 * self.config_class.V_MAX + 1e-9), -1.0, 1.0)
			# Nearest-neighbor distance normalized by separation distance
			nn_dist_norm = np.clip(nn_dist / (3.0 * self.separation_distance + 1e-9), 0.0, 2.0)
		else:
			closing_rate_norm = 0.0
			nn_dist_norm = 2.0  # no neighbor → "far away"

		# Rotate each neighbor vector into ego frame, then flatten to 4 slots
		rotated_neighbors = [
			get_rotated_position_from_relative(np.asarray(vec, dtype=np.float32), agent_heading).astype(np.float32)
			for vec in nearest
		]

		nearest_neighbors = np.concatenate(rotated_neighbors, axis=0)


		# --- Tube params (rotated entrance/exit vectors + width + phase) ---
		# tube_entrance = np.asarray(world.tube_params['entrance'], dtype=np.float32)
		# tube_exit = np.asarray(world.tube_params['exit'], dtype=np.float32)
		# rel_to_exit_world = tube_exit - agent_pos
		# rot_rel_exit = get_rotated_position_from_relative(rel_to_exit_world, agent_heading).astype(np.float32)

		# tube_exit = np.asarray(world.tube_params['exit'], dtype=np.float32)
		# tube_width = float(world.tube_params['width'])

		# Phase is computed in world coords; keep as scalar to preserve layout
		phase = float(self.get_agent_phase(agent, world))

		s, y, L, half_w = self._tube_coords(world, agent_pos, self.current_tube[agent.id])
		s_norm = np.clip(s / L, -2.0, 2.0)          # allow slight overshoot
		y_norm = np.clip(y / (half_w + 1e-9), -2.0, 2.0)
		dist_in = self._entrance_gate_distance(s, y, half_w) / (L + 1e-9)
		dist_out = self._exit_gate_distance(s, y, L, half_w) / (L + 1e-9)
		# print("Agent", agent.id, "s,y,L,half_w:", s_norm, y_norm, "dist_in:", dist_in, "dist_out:", dist_out)
		# === NEW: Add heading alignment feature ===
		current_tube = world.tube_params[self.current_tube[agent.id]]
		corridor_vec = current_tube['e']
		corridor_heading = np.arctan2(corridor_vec[1], corridor_vec[0])
		heading_error = (agent_heading - corridor_heading + np.pi) % (2*np.pi) - np.pi
		# heading_alignment = np.array([np.cos(heading_error), np.sin(heading_error)], dtype=np.float32)
		tube_params = np.concatenate([
			np.array([s_norm, y_norm]),                                       # tube-frame position
			np.array([dist_in, dist_out], dtype=np.float32),                  # distance to entrance & exit
			np.array([closing_rate_norm, nn_dist_norm, phase], dtype=np.float32)  # NN closing rate, NN dist, phase
		], axis=0)
		# print("Agent", agent.id, "tube_params", tube_params, "np.array([agent.state.speed,agent_speed])", np.array([agent.state.speed, agent_speed]))

		# print("Agent", agent.id, "tube coords s,y,L,half_w:", s, y, L, half_w)
		# --- Assemble final obs in the SAME field order as before ---
		# [agent_vel(2), goal_pos(2), nearest_neighbors(4), tube_params(8)] = 16 dims
		return np.concatenate([
			np.array([np.cos(heading_error), np.sin(heading_error), agent.state.speed]),  # corridor-relative heading + speed
			goal_pos,                           # rotated goal vector (ego frame)
			nearest_neighbors,                  # two rotated neighbor vectors (ego frame)
			tube_params                         # tube-frame features
		], axis=0).astype(np.float32)

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
			# 	print("agent_done", entity.id, disconnected)
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



	def _get_entity_feat_relative(self, agent: Agent, entity: Entity, world: World, fairness_param: np.ndarray) -> arr:
		"""
		Returns rotation-invariant node features for `entity` relative to `agent`.

		Agents/Landmarks/Obstacles:
			[rel_vel(2), rel_pos(2), rel_goal_pos(2), goal_occupied(1), entity_type(1)]

		Walls:
			[rel_vel(2), rel_pos(2), rel_goal_pos(2), goal_occupied(1),
			goal_history(1), wall_o_corner(2), wall_d_corner(2), entity_type(1)]
		"""
		# --- Ego (reference) state ---
		agent_pos = np.asarray(agent.state.p_pos, dtype=np.float32)
		agent_vel = np.asarray(agent.state.p_vel, dtype=np.float32)
		agent_heading = float(getattr(agent.state, "theta", getattr(agent.state, "p_ang", 0.0)))

		# --- Entity relative vectors in WORLD frame ---
		entity_pos_world = np.asarray(entity.state.p_pos, dtype=np.float32)
		entity_vel_world = np.asarray(entity.state.p_vel, dtype=np.float32)
		rel_pos_world = entity_pos_world - agent_pos
		rel_vel_world = entity_vel_world - agent_vel

		# --- Rotate into ego (agent) frame ---
		rel_pos = get_rotated_position_from_relative(rel_pos_world, agent_heading).astype(np.float32)
		rel_vel = get_rotated_position_from_relative(rel_vel_world, agent_heading).astype(np.float32)

		if 'agent' in entity.name:
			# Each agent's goal is the matched landmark pose
			goal_pos_world = np.asarray(self.landmark_poses[self.goal_match_index[entity.id]], dtype=np.float32)
			## change the goal to the exit position of the corridor
			# goal_pos_world = np.asarray(world.tube_params['exit'], dtype=np.float32)
			rel_goal_pos_world = goal_pos_world - agent_pos
			rel_goal_pos = get_rotated_position_from_relative(rel_goal_pos_world, agent_heading).astype(np.float32)

			entity_type = np.array([entity_mapping['agent']], dtype=np.float32)

			return np.hstack([rel_vel, rel_pos, rel_goal_pos, entity_type]).astype(np.float32)

		elif 'landmark' in entity.name:
			# For landmarks, the "goal" is its own position relative to ego
			rel_goal_pos = rel_pos.copy()
			entity_type = np.array([entity_mapping['landmark']], dtype=np.float32)

			return np.hstack([rel_vel, rel_pos, rel_goal_pos, entity_type]).astype(np.float32)

		elif 'obstacle' in entity.name:
			# Same layout as landmarks
			rel_goal_pos = rel_pos.copy()
			entity_type = np.array([entity_mapping['obstacle']], dtype=np.float32)

			return np.hstack([rel_vel, rel_pos, rel_goal_pos, entity_type]).astype(np.float32)

		elif 'wall' in entity.name:
			# For walls, include rotated corner points as you already did
			rel_goal_pos = rel_pos.copy()
			goal_history = np.array([entity.id if entity.id is not None else 0], dtype=np.float32)
			entity_type = np.array([entity_mapping['wall']], dtype=np.float32)

			# Compute wall corners in WORLD frame, then rotate into ego frame
			# (Assumes your wall encodes a segment via endpoints[] along 'axis_pos' with a width)
			wall_o_corner_world = np.array([entity.endpoints[0], entity.axis_pos + entity.width / 2.0], dtype=np.float32) - agent_pos
			wall_d_corner_world = np.array([entity.endpoints[1], entity.axis_pos - entity.width / 2.0], dtype=np.float32) - agent_pos
			wall_o_corner = get_rotated_position_from_relative(wall_o_corner_world, agent_heading).astype(np.float32)
			wall_d_corner = get_rotated_position_from_relative(wall_d_corner_world, agent_heading).astype(np.float32)

			return np.hstack([
				rel_vel, rel_pos, rel_goal_pos, goal_history, wall_o_corner, wall_d_corner, entity_type
			]).astype(np.float32)

		else:
			raise ValueError(f'{entity.name} not supported')



# actions: [None, ←, →, ↓, ↑, comm1, comm2]
if __name__ == "__main__":

	from multiagent.environment import MultiAgentGraphEnv
	from multiagent.policy import InteractivePolicy

	# makeshift argparser
	class Args:
		def __init__(self):
			self.num_agents:int=3
			self.world_size=6
			self.num_scripted_agents=0
			self.num_obstacles:int=3
			self.collaborative:bool=False 
			self.max_speed:Optional[float]=2
			self.collision_rew:float=5
			self.goal_rew:float=50
			self.min_dist_thresh:float=0.1
			self.use_dones:bool=True
			self.episode_length:int=200
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