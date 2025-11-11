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

		world.positions_all_agents1 = []
		world.positions_all_agents2 = []
		world.positions_all_agents0 = []
		world.positions_all_agents3 = []
		world.positions_all_agents4 = []
		world.positions_all_agents5 = []
		world.positions_all_agents6 = []
		world.positions_all_agents7 = []
		world.positions_all_agents8 = []
		world.positions_all_agents9 = []

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
		self.prev_goal_dist = np.full(self.num_agents, np.inf, dtype=np.float32)


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

		self.agent_dist_traveled = np.zeros(self.num_agents)
		self.agent_time_taken = np.zeros(self.num_agents)
		wall_length = np.random.uniform(0.2, 0.8)
		self.wall_length = wall_length * self.world_size/4

		self.phase_reached = np.zeros(self.num_agents)
		self.entry_reward_cooldown = np.zeros(self.num_agents, dtype=np.int32)
		# Store previous longitudinal position (s) for progress reward
		self.prev_proj = np.zeros(self.num_agents, dtype=np.float32)

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
		# Initialize tube parameters
		self.setup_tube_params(world)
		num_agents_added = 0
		agents_added = []
		boundary_thresh = 0.9

		while True:
			if num_agents_added == self.num_agents:
				break
			# Random position in top half of world
			# random_pos = boundary_thresh * np.random.uniform(
			# 	[-self.world_size,self.world_size/4],  # min x,y
			# 	[self.world_size, self.world_size],  # max x,y
			# 	world.dim_p
			# )
			# Add random jitter if needed
			jitter = 0.3 * np.random.uniform(-self.world_size, self.world_size, world.dim_p)
			angle = world.tube_params['angle']
			# print("jitter", jitter)
			perp_dir = np.array([np.sin(angle), np.cos(angle)])
			# print("Entrance", world.tube_params['entrance'], "perp_dir", perp_dir )
			# print("self.world_size+(num_agents_added) / 5 * perp_dir",( self.world_size+(num_agents_added)) / 5 * perp_dir)
			distance_from_entrance = (self.world_size + num_agents_added) / 3
			# print("distance_from_entrance", distance_from_entrance, "jitter", jitter)
			random_pos = world.tube_params['entrance'] + distance_from_entrance * perp_dir + jitter
			# print(f"Random Position for Agent {num_agents_added}: {random_pos}")
			# input("Press Enter to continue...")
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
		# agent_pos = [agent.state.p_pos for agent in world.agents]
		#####################################################


		self.agent_id_updated = np.arange(self.num_agents)
		if self.formation_type == 'line':
			set_landmarks_in_line(self, world, line_angle=0, start_pos=np.array([-self.world_size/2, -self.world_size/2]), end_pos=np.array([self.world_size/2,-self.world_size/2]))
		elif self.formation_type == 'circle':
			set_landmarks_in_circle(self, world, center=np.array([0.0, world.tube_params['exit'][1]+self.world_size/5]), radius=self.world_size/3)
		elif self.formation_type == 'point':
			set_landmarks_in_point(self, world, tube_angle=world.tube_params['angle'], tube_endpoints=world.tube_params['exit'])
		# elif self.formation_type == 'random':
		# 	set_landmarks_random(self, world)
		# else:
		# 	raise NotImplementedError

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


	def setup_tube_params(self, world):
		"""
		Set up tube parameters using modified landmark line logic
		"""
		# Initialize tube list
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
		self.exit_back_ratio = getattr(self, 'exit_back_ratio', 0.05)    # inside tube near exit
		self.exit_front_ratio = getattr(self, 'exit_front_ratio', 0.08)  # just outside exit	

	# --- Shared geometry helpers (reduce redundancy) ---
	def _tube_frame(self, world: World):
		tp = world.tube_params
		return tp['entrance'], tp['e'], tp['n'], float(tp['L']), float(tp['half_width'])

	def _tube_coords(self, world: World, pos: np.ndarray):
		"""Return (s, y, L, half_w): longitudinal s from entrance and signed lateral y."""
		entrance, e, n, L, half_w = self._tube_frame(world)
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

	# def is_in_tube(self, world: World, pos):
	# 	"""
	# 	Updated helper function to check if position is inside rotated tube
	# 	"""
	# 	entrance = world.tube_params['entrance']
	# 	exit = world.tube_params['exit']
	# 	width = world.tube_params['width']
	# 	angle = world.tube_params['angle']
		
	# 	# Calculate tube direction vector
	# 	tube_direction = exit - entrance
	# 	tube_length = np.linalg.norm(tube_direction)
	# 	tube_direction = tube_direction / tube_length
		
	# 	# Calculate vector from entrance to position
	# 	pos_vector = pos - entrance
		
	# 	# Project position onto tube direction
	# 	projection_length = np.dot(pos_vector, tube_direction)
		
	# 	# Check if point is between entrance and exit
	# 	if projection_length < 0 or projection_length > tube_length:
	# 		return False
			
	# 	# Calculate perpendicular distance to tube centerline
	# 	perpendicular_vector = pos_vector - projection_length * tube_direction
	# 	perpendicular_distance = np.linalg.norm(perpendicular_vector)
		
	# 	# Check if point is within tube width
	# 	return perpendicular_distance <= width/2

	# --- Optimized queries using the frame ---
	def is_in_tube(self, world: World, pos: np.ndarray) -> bool:
		s, y, L, half_w = self._tube_coords(world, pos)
		return self._in_tube_rect(s, y, L, half_w)
	
	def get_agent_phase(self, agent: Agent, world: World):
		pos = agent.state.p_pos
		# in_tube = self.is_in_tube(world, pos)
		s, y, L, half_w = self._tube_coords(world, pos)
		in_tube = self._in_tube_rect(s, y, L, half_w)
		# print("Agent", agent.id, "position", pos, "in_tube:", in_tube)
		passed_tube = (s > L)
		valid_entrance = self._in_entrance_gate(s, y, L, half_w)
		valid_exit = self._in_exit_gate(s, y, L, half_w)
		# print("Passed tube1 :", passed_tube, "valid_entrance:", valid_entrance, "valid_exit:", valid_exit)

		# exit_to_pos = pos - world.tube_params['exit']
		# tube_direction = world.tube_params['exit'] - world.tube_params['entrance']
		# tube_direction = tube_direction / np.linalg.norm(tube_direction)
		# passed_tube = np.dot(exit_to_pos, tube_direction) > 0
		# entrance_to_pos = pos - world.tube_params['entrance']
		# proj = np.dot(entrance_to_pos, tube_direction)
		# perp_dist = np.linalg.norm(entrance_to_pos - proj * tube_direction)
		# tube_width = world.tube_params['width']
		# tube_length = np.linalg.norm(world.tube_params['exit'] - world.tube_params['entrance'])

		# valid_entrance = (0 <= proj < 0.1 * tube_length) and (perp_dist < tube_width / 2)
		# valid_entrance = self._in_entrance_gate(s, y, L, half_w)

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


	# def _bipartite_min_dists(self, dists):
	# 	ri, ci = linear_sum_assignment(dists)
	# 	min_dists = dists[ri, ci]
	# 	return min_dists


	def reward(self, agent: Agent, world: World) -> float:
		rew = 0.0
		current_phase = self.get_agent_phase(agent, world)
		# print("Agent", agent.id, "phase", current_phase, "previous_phase", agent.previous_phase, "phase_reached", self.phase_reached[agent.id])
		# print("Goalrew",self.goal_rew, "Collisionrew",self.collision_rew)
		# Common rewards across all phases
		# Collision penalties
		if agent.collide:
			for a in world.agents:
				if a.id == agent.id:
					continue
				if self.is_collision(a, agent):
					rew -= self.collision_rew*4
					# print(f"!!!Agent {agent.id} collided with agent {a.id} penalty",self.collision_rew*4 )
					# print(" self.separation_distance", self.separation_distance)
					# input("Collision")
			
			if self.is_obstacle_collision(pos=agent.state.p_pos,
										entity_size=agent.size, 
										world=world):
				rew -= self.collision_rew*3
				# print(f"Agent {agent.id} collided with obstacle")

		# Calculate tube length
		tube_direction = world.tube_params['exit'] - world.tube_params['entrance']
		tube_length = np.linalg.norm(tube_direction)
		agent_pos = agent.state.p_pos
		agent_heading = agent.state.theta
		heading_vec = np.array([np.cos(agent_heading), np.sin(agent_heading)])
		s, y, L, half_w = self._tube_coords(world, agent_pos)
		front_agents = []
		back_agents = []
		# print("self._in_entrance_gate(s, y, L, half_w)",self._in_entrance_gate(s, y, L, half_w))
		for other in world.agents:
			if other is agent:
				continue
			rel_vec = other.state.p_pos - agent_pos
			proj = np.dot(rel_vec, heading_vec)
			if proj > 0:
				front_agents.append((proj, other))
			else:
				back_agents.append((proj, other))

		# Get closest in front and back
		front_agent = min(front_agents, key=lambda x: x[0])[1] if front_agents else None
		back_agent = max(back_agents, key=lambda x: x[0])[1] if back_agents else None

		# print("front_agent", front_agent.id if front_agent else "None")
		# print("back_agent", back_agent.id if back_agent else "None")
		
		# Calculate desired spacing based on tube length and number of agents
		desired_spacing = self.separation_distance  # 3 is the number of agents in the tube TODO: harcoded

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
			rew -= self.goal_rew  #*3  # Reduced penalty
			# print(f"Agent {agent.id} penalized for skipping from phase {agent.previous_phase} to {current_phase} rew", rew)
		tube_direction_vector = tube_direction / tube_length
		entrance_to_agent = agent.state.p_pos - world.tube_params['entrance']
		proj = np.dot(entrance_to_agent, tube_direction_vector)
		# entrance_dist = np.linalg.norm(entrance_to_agent - proj * tube_direction)

		if current_phase == agent.previous_phase+1 and self.phase_reached[agent.id] == current_phase-1:
			# print("HELLO",agent.id, "current_phase", current_phase, "previous_phase", agent.previous_phase)
			# print("proj", proj," 0.1 * tube_length", 0.1 * tube_length)
			# print("entrance_dist", entrance_dist, " 0.2* tube_length", 0.2 * tube_length)
			# Reward proper phase progression
			# if current_phase == 1 and 0 <= proj < 0.1 * tube_length and entrance_dist <  0.2 * tube_length and (self.entry_reward_cooldown[agent.id] == 0):
			if current_phase == 1 and self._in_entrance_gate(s, y, L, half_w) and (self.entry_reward_cooldown[agent.id] == 0):
				# print("Agent pos, entrance, exit", agent.state.p_pos, world.tube_params['entrance'], world.tube_params['exit'])
				# print("distance check", proj, entrance_dist)

				# Reward if agent moves into tube after exiting
				rew += self.goal_rew  # *3  # Positive reward for proper transition
				self.entry_reward_cooldown[agent.id] = self.phase_reward_cooldown_steps  # Cooldown period to prevent repeated rewards
				self.phase_reached[agent.id] = 1  # Mark Phase 1 completed
				# print(f"Agent {agent.id} properly progressed from phase {agent.previous_phase} to {current_phase} rew", rew)
			elif current_phase == 2 :
				# Rewards if agent moves out of tube
				# print("Agent in post-tube phase", agent.id)
				rew += self.goal_rew  # *3
				self.phase_reached[agent.id] = 2  # Mark Phase 2 completed
				# # print("Agent",agent.id,"reached fair goal")
				# if agent.status == False:
				# 	agent.status = True
				# 	agent.state.reset_velocity()
				# print(f"Agent {agent.id} properly progressed from phase {agent.previous_phase} to {current_phase} rew", rew)
				# Update the global phase tracker if any agent progresses

		# Phase-specific rewards
		# print("Agent",agent.id,"current_phase",current_phase,"prev phase_reached",self.phase_reached)
		if current_phase == 0:  # Pre-tube phase
			# Reward for getting closer to tube entrance
			# dist_to_entrance = np.linalg.norm(world.tube_params['entrance'] - agent.state.p_pos)
			# rew -= dist_to_entrance
			# print("Agent", agent.id, " Phase 0 dist_to_entrance", dist_to_entrance, "rew", rew)
			s, y, L, half_w = self._tube_coords(world, agent.state.p_pos)
			# print("Agent", agent.id, " Phase 0 s,y,L,half_w:", s, y, L, half_w)
			dist_to_entrance_edge = self._entrance_gate_distance(s, y, half_w)
			rew -= dist_to_entrance_edge
			# print("Agent", agent.id, " Phase 0 dist_to_entrance_edge", dist_to_entrance_edge, "rew", rew)
			# # Formation reward considering front and back agents
			# spacing_error = 0
			# if front_agent:
			# 	diff = np.linalg.norm(front_agent.state.p_pos - agent.state.p_pos) - desired_spacing
			# 	spacing_error += np.abs(diff) if diff < 0 else 0
			# if back_agent:
			# 	diff = np.linalg.norm(back_agent.state.p_pos - agent.state.p_pos) - desired_spacing
			# 	spacing_error += np.abs(diff) if diff < 0 else 0
			# rew -= spacing_error *  self.formation_rew
			# print("Phase 0 spacing_error",spacing_error)
			# input("phase 0")

		elif current_phase == 1:  # In-tube phase
			# print("Agent", agent.id, "in tube phase 1")
			# print("formation line",self.formation_rew)
			# rew += self.formation_rew/2  # Reward for entering tube
			# Stronger formation rewards inside tube
			spacing_error = 0
			max_spacing_error = 0


			if front_agent:
				# print("np.linalg.norm(front_agent.state.p_pos - agent.state.p_pos)", np.linalg.norm(front_agent.state.p_pos - agent.state.p_pos), "desired_spacing", desired_spacing)
				diff = np.linalg.norm(front_agent.state.p_pos - agent.state.p_pos) - desired_spacing
				# print("diff", diff)
				spacing_error += np.abs(diff) if diff < 0 else 0
				max_spacing_error = max(max_spacing_error, np.abs(diff))
			if back_agent:
				# print("np.linalg.norm(back_agent.state.p_pos - agent.state.p_pos)", np.linalg.norm(back_agent.state.p_pos - agent.state.p_pos), "desired_spacing", desired_spacing)
				diff = np.linalg.norm(back_agent.state.p_pos - agent.state.p_pos) - desired_spacing
				# print("diff", diff)
				max_spacing_error = max(max_spacing_error, np.abs(diff))
				spacing_error += np.abs(diff) if diff < 0 else 0
			if spacing_error > 0:
				# print("Phase 1 spacing_error",spacing_error)
				self.spacing_violation[agent.id] += 1
			rew -= spacing_error *  self.formation_rew  # Higher weight for maintaining formation in tube
			# print("Phase 1 spacing_error", spacing_error)

			# print("Phase 1 spacing_error",max_spacing_error)
			# Progress through tube
			# dist_to_exit = np.linalg.norm(world.tube_params['exit'] - agent.state.p_pos)
			# rew -= dist_to_exit
			dist_to_exit_edge = self._exit_gate_distance(s, y, L, half_w)
			rew -= dist_to_exit_edge
			# print("Phase1 Agent", agent.id, "dist_to_exit_edge",dist_to_exit_edge, "rew", rew)
			delta_proj = proj - float(self.prev_proj[agent.id])
			# Positive reward for forward progress
			rew += self.progress_gain * max(delta_proj, -0.05)  # clamp small negative drift
			# print("Phase1 Agent",agent.id,"delta_proj",delta_proj, "rew", rew)
			# print("dist_to_exit",dist_to_exit, "rew", rew)
			# print("Agent",agent.id,"Phase 1 spacing_error",spacing_error)
			self.delta_spacing.append(spacing_error)
			self.steps_in_corridor[agent.id] += 1
			self.prev_proj[agent.id] = proj  # update after using delta_s

			# print("delta_spacing",self.delta_spacing)
			# print("Agent",agent.id,"delta_spacing",self.delta_spacing[agent.id])
			# input("phase 1")
		
		elif current_phase == 2 and self.phase_reached[agent.id] == 0:  # Post-tube phase
			# print("Agent",agent.id,"post tube phase")
			# input("Agent entered post tube phase")
			current_phase = 0  # Reset current phase to 0
		elif current_phase == 2:  # Post-tube phase
			dist_to_goal = np.linalg.norm(agent.state.p_pos - self.landmark_poses[self.goal_match_index[agent.id]])
			if dist_to_goal < self.min_dist_thresh:
				# print("Agent",agent.id,"reached fair goal")
				if agent.status is False:
					agent.status = True
					agent.state.reset_velocity()
					rew += self.goal_rew*5
					self.goal_tracker[agent.id] = self.goal_match_index[agent.id]

					# print("Phase 2 Agent", agent.id, "reached goal", dist_to_goal, "rew", rew)
			else:
				rew -= dist_to_goal
				# print("Still dist_to_goal", dist_to_goal, "rew", rew)
				# print("Agent", agent.id, "not reached goal", dist_to_goal)
				# input("Not reached goal yet")
				# Reward forward progress toward goal
				# if hasattr(self, 'prev_goal_dist'):
				# 	delta_goal = self.prev_goal_dist[agent.id] - dist_to_goal
				# 	rew += self.progress_gain * max(delta_goal, -0.1) * 2.0  # 2x boost in Phase 2
				
				# if not hasattr(self, 'prev_goal_dist'):
				# 	self.prev_goal_dist = np.full(self.num_agents, np.inf, dtype=np.float32)
				# self.prev_goal_dist[agent.id] = dist_to_goal
			# input("phase 2")

			# spacing_error = 0
			# if front_agent:
			# 	diff = np.linalg.norm(front_agent.state.p_pos - agent.state.p_pos) - desired_spacing
			# 	spacing_error += np.abs(diff) if diff < 0 else 0
			# if back_agent:
			# 	diff = np.linalg.norm(back_agent.state.p_pos - agent.state.p_pos) - desired_spacing
			# 	spacing_error += np.abs(diff) if diff < 0 else 0
			# rew -= spacing_error *  self.formation_rew # Higher weight for maintaining formation in tube

		# # # Global formation quality (calculated once per step)
		# if agent.id == 0:
		# 	all_spacings = []
		# 	for a in world.agents:
		# 		a_neighbor_dists = [np.linalg.norm(other.state.p_pos - a.state.p_pos) 
		# 						for other in world.agents if other is not a]
		# 		if len(a_neighbor_dists) >= 2:
		# 			a_neighbor_dists.sort()
		# 			all_spacings.extend(a_neighbor_dists[:2])
		# 	# print("all_spacings",all_spacings)
		# 	spacing_std = np.std(all_spacings)
		# 	# self.spacing_min = np.min(all_spacings)
		# 	# print("Global spacing_std",spacing_std*self.formation_rew )
		# 	rew -= spacing_std *  self.formation_rew  # Reward uniform spacing across all agents




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
		if self._in_tube_rect(s, y, L, half_w) and not current_phase == 1:
			rew -= self.collision_rew  #*2
			# print(f"Agent {agent.id} is in tube but not in phase 1 rew", rew)

        # If agent is past exit plane but never entered corridor: continuous penalty
		if s > L and self.phase_reached[agent.id] < 1:
			rew -= self.goal_rew
			# print(f"Agent {agent.id} skipped corridor (s={s:.2f} > L={L:.2f}): penalty {self.goal_rew}")
        
		# print(f"Agent {agent.id} total reward: ", rew)
		# input("Reward calculation complete for agent {}".format(agent.id))
		# input("Press Enter to continue...")

		return np.clip(rew, -4*self.collision_rew, self.goal_rew*5)


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
		closest_goal_occupied = goal_occupied  ##np.array([self.landmark_poses_occupied[top_two_indices[0]]])
		# second_closest_goal_occupied = ## np.array([self.landmark_poses_occupied[top_two_indices[1]]])
		# input("Press Enter to continue...")
		return goal_pos, closest_goal_occupied, rel_second_closest_goal, goal_history



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

		# --- Two nearest neighbors (rotated) ---
		neighbor_dists = []
		for other in world.agents:
			if other is agent:
				continue
			rel_pos_world = other.state.p_pos - agent_pos
			dist = float(np.linalg.norm(rel_pos_world))
			neighbor_dists.append((dist, rel_pos_world))

		neighbor_dists.sort(key=lambda x: x[0])
		nearest = [n[1] for n in neighbor_dists[:2]]
		while len(nearest) < 2:
			nearest.append(np.zeros(world.dim_p, dtype=np.float32))

		# Rotate each neighbor vector into ego frame, then flatten to 4 slots
		rotated_neighbors = [
			get_rotated_position_from_relative(np.asarray(vec, dtype=np.float32), agent_heading).astype(np.float32)
			for vec in nearest
		]

		nearest_neighbors = np.concatenate(rotated_neighbors, axis=0)

		# --- Tube params (rotated entrance/exit vectors + width + phase) ---
		# tube_entrance = np.asarray(world.tube_params['entrance'], dtype=np.float32)
		# tube_exit = np.asarray(world.tube_params['exit'], dtype=np.float32)
		tube_width = float(world.tube_params['width'])

		# rel_to_entrance_world = tube_entrance - agent_pos
		# rel_to_exit_world = tube_exit - agent_pos

		# rot_rel_entrance = get_rotated_position_from_relative(rel_to_entrance_world, agent_heading).astype(np.float32)
		# rot_rel_exit = get_rotated_position_from_relative(rel_to_exit_world, agent_heading).astype(np.float32)


		# Phase is computed in world coords; keep as scalar to preserve layout
		phase = float(self.get_agent_phase(agent, world))

		# # Heading error to corridor axis (axis vector e)
		# e = world.tube_params['e']
		# tube_axis_angle = float(np.arctan2(e[1], e[0]))

		# # print("Agent", agent.id, "tube_axis_angle", (tube_axis_angle+np.pi)%(2*np.pi), "angle", (world.tube_params['angle'] + np.pi/2)%(2*np.pi), "agent_heading", (agent_heading+np.pi)%(2*np.pi))
		# dpsi = agent_heading - tube_axis_angle
		# # Wrap to [-pi, pi]
		# dpsi = (dpsi + np.pi) % (2 * np.pi) - np.pi
		# heading_feat = np.array([np.cos(dpsi), np.sin(dpsi)], dtype=np.float32)
		# # print("Agent", agent.id, "heading_feat", heading_feat)

		s, y, L, half_w = self._tube_coords(world, agent_pos)
		s_norm = np.clip(s / L, -2.0, 2.0)          # allow slight overshoot
		y_norm = np.clip(y / (half_w + 1e-9), -2.0, 2.0)
		dist_in = self._entrance_gate_distance(s, y, half_w) / (L + 1e-9)
		dist_out = self._exit_gate_distance(s, y, L, half_w) / (L + 1e-9)
		# print("Agent", agent.id, "s,y,L,half_w:", s_norm, y_norm, "dist_in:", dist_in, "dist_out:", dist_out)

		tube_params = np.concatenate([
			np.array([s_norm, y_norm]),  # rot_rel_entrance,
			# np.array([dist_out], dtype=np.float32),  # rot_rel_exit, dist_in, 
			# heading_feat,                  
			# np.array([tube_width], dtype=np.float32),
			np.array([phase], dtype=np.float32)
		], axis=0)
		# print("Agent", agent.id, "tube_params", tube_params, "np.array([agent.state.speed,agent_speed])", np.array([agent.state.speed, agent_speed]))

		# print("Agent", agent.id, "tube coords s,y,L,half_w:", s, y, L, half_w)
		# --- Assemble final obs in the SAME field order as before ---
		# [agent_vel(2), goal_pos(2), nearest_neighbors(4), tube_params(8)] = 16 dims
		return np.concatenate([
			np.array([np.cos(agent.state.theta), np.sin(agent.state.theta), agent.state.speed]), # np.array([np.cos(agent.state.theta), np.sin(agent.state.theta), agent.state.speed]),		#   rot_agent_vel, # self velocity (2 slots)
			goal_pos,                           # rotated goal vector
			nearest_neighbors,                  # two rotated neighbor vectors
			tube_params                         # rotated entrance/exit + width + phase
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
