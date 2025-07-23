class Scenario(BaseScenario):
	def make_world(self, args:argparse.Namespace) -> World:
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

	def reset_world(self, world:World, num_current_episode: int = 0) -> None:
		# print("RESET WORLD")
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

			agent.state.p_dist = 0.0
			agent.state.time = 0.0
		# set colours for scripted agents
		for i, agent in enumerate(world.scripted_agents):
			agent.color = np.array([0.15, 0.15, 0.15])
		# set colours for landmarks
		for i, landmark in enumerate(world.landmarks):
			landmark.color = np.array([0.85, 0.35, 0.35])

		#####################################################
		# self.update_curriculum(world, num_current_episode)
		self.random_scenario(world)
		self.initialize_min_time_distance_graph(world)
		
	def get_rotated_position_from_relative(relative_position: np.ndarray,
		reference_heading: float):
		# returns relative position from the reference state.
		assert relative_position.shape == (2,), "relative_position should be a 2D array."
		rot_matrix = np.array([[np.cos(reference_heading), np.sin(reference_heading)], [-np.sin(reference_heading), np.cos(reference_heading)]])
		relative_position_rotated = np.dot(rot_matrix, relative_position)
		return relative_position_rotated
	
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
				[-self.world_size, 0],  # min x,y
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
			set_landmarks_in_line(self, world, line_angle=0, start_pos=np.array([-self.world_size/1.5, -self.world_size/1.5]), end_pos=np.array([self.world_size/1.5,-self.world_size/1.5]))
		elif self.formation_type == 'circle':
			set_landmarks_in_circle(self, world, center=np.array([0.0, world.tube_params['exit'][1]+self.world_size/5]), radius=self.world_size/3)
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
		
	def setup_tube_params(self, world):
		"""
		Set up tube parameters using modified landmark line logic
		"""
		# Calculate tube width based on number of agents
		self.tube_width = max(
			self.num_agents * world.agents[0].size * 2.5,  # Width based on agents
			self.world_size * 0.15  # Minimum width
		)
		
		# Calculate tube entrance (middle of world)
		entrance_y = self.world_size/2 * 0.5  # Middle of world
		entrance_x = 0  # Center horizontally
		
		# Calculate tube exit (bottom of world)
		exit_y = -self.world_size/2 * 0.5  # Near bottom of world
		exit_x = 0  # Same x as entrance
		
		# Store tube parameters
		world.tube_params = {
			'entrance': np.array([entrance_x, entrance_y]),
			'exit': np.array([exit_x, exit_y]),
			'width': self.tube_width,
			'angle': np.pi/2  # Vertical tube
		}

		# Calculate line formation reference points
		line_length = self.tube_width * 0.8  # Slightly smaller than tube width
		
		# Pre-tube formation line (above entrance)
		pre_tube_y = entrance_y + self.world_size * 0.2
		self.pre_tube_line = {
			'start_pos': np.array([-line_length/2, pre_tube_y]),
			'end_pos': np.array([line_length/2, pre_tube_y]),
			'angle': 0  # Horizontal line
		}
		
		# Post-tube target line (at top of world)
		post_tube_y = self.world_size/2 * 0.8
		self.post_tube_line = {
			'start_pos': np.array([-line_length/2, post_tube_y]),
			'end_pos': np.array([line_length/2, post_tube_y]),
			'angle': 0  # Horizontal line
		}
		
		# Store additional parameters that might be useful
		world.tube_params.update({
			'length': entrance_y - exit_y,
			'pre_tube_line': self.pre_tube_line,
			'post_tube_line': self.post_tube_line
		})


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
		return collision
	
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



		if world.dists_to_goal[agent.id] == -1:
			mean_dist,  std_dev_dist, _ = self.collect_dist(world)
			fairness_param = mean_dist/(std_dev_dist+0.0001)
		else:
			fairness_param = world.dist_traveled_mean/(world.dist_traveled_stddev+0.0001)

			scaled_input = fairness_param
			tanh_output = np.tanh(scaled_input-self.zeroshift)
			fair_rew = self.fair_rew * tanh_output
			# reduce the negative reward if the fairness is not met
			if fair_rew < -self.fair_rew:
				fair_rew = -self.fair_rew

				
			rew += fair_rew 

		if agent.id == 0:

			##### fair assignment to check if it makes it more fair
			agent_pos = [agent.state.p_pos for agent in world.agents]
			costs = dist.cdist(agent_pos, self.landmark_poses)


			x, objs = solve_fair_assignment(costs)
			self.goal_match_index = np.where(x==1)[1]
			# print("Goal Match Index",self.goal_match_index)
			if self.goal_match_index.size == 0 or self.goal_match_index.shape != (self.num_agents,):
				self.goal_match_index = np.arange(self.num_agents)

		# Common rewards across all phases
		# Collision penalties
		if agent.collide:
			for a in world.agents:
				if a.id == agent.id:
					continue
				if self.is_collision(a, agent):
					rew -= self.collision_rew*3
			
			if self.is_obstacle_collision(pos=agent.state.p_pos,
										entity_size=agent.size, 
										world=world):
				rew -= self.collision_rew*3

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
		
		# Track agent's previous phase if not already stored
		if not hasattr(agent, 'previous_phase'):
			agent.previous_phase = 0
		
		# ##Modified phase transition handling
		if current_phase == 2 and current_phase > agent.previous_phase + 1:
			# Only penalize clear phase skips (e.g., 0 to 2)
			rew -= self.goal_rew*3  # Reduced penalty
		if current_phase == agent.previous_phase+1 and self.phase_reached[agent.id] == current_phase-1:
			# Reward proper phase progression
			if current_phase == 1 and agent.state.p_pos[1] >= (world.tube_params['entrance'][1]+world.tube_params['exit'][1])/2:
				# Reward if agent moves into tube after exiting
				rew += self.goal_rew*3  # Positive reward for proper transition
			elif current_phase == 2 :
				# Rewards if agent moves out of tube
				rew += self.goal_rew*3
				# Update the global phase tracker if any agent progresses
		if current_phase > self.phase_reached[agent.id]:
			self.phase_reached[agent.id] = current_phase  # Update max phase reached globally
		# Store current phase for next step
		agent.previous_phase = current_phase
		# Phase-specific rewards
		if current_phase == 0:  # Pre-tube phase
			# Reward for getting closer to tube entrance
			dist_to_entrance = np.linalg.norm(world.tube_params['entrance'] - agent.state.p_pos)
			rew -= dist_to_entrance

			# Formation reward considering front and back agents
			spacing_error = 0
			if front_agent:
				spacing_error += np.abs(np.linalg.norm(front_agent.state.p_pos - agent.state.p_pos) - desired_spacing)
			if back_agent:
				spacing_error += np.abs(np.linalg.norm(back_agent.state.p_pos - agent.state.p_pos) - desired_spacing)
			rew -= spacing_error *  self.formation_rew
				
		elif current_phase == 1:  # In-tube phase
			# Stronger formation rewards inside tube
			spacing_error = 0
			if front_agent:
				spacing_error += np.abs(np.linalg.norm(front_agent.state.p_pos - agent.state.p_pos) - desired_spacing)
			if back_agent:
				spacing_error += np.abs(np.linalg.norm(back_agent.state.p_pos - agent.state.p_pos) - desired_spacing)
			rew -= spacing_error *  self.formation_rew # Higher weight for maintaining formation in tube
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
					rew += self.goal_rew*5

			else:
					rew -= dist_to_fair_goal
		
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
			# print("Global spacing_std",spacing_std)
			rew -= spacing_std *  self.formation_rew  # Reward uniform spacing across all agents
		# print("Agent",agent.id,"rew",rew)

		return np.clip(rew, -2*self.collision_rew, self.goal_rew)

	def is_in_tube(self, world: World, pos):
		"""Helper function to check if position is inside tube"""
		tube_entrance = world.tube_params['entrance']
		tube_exit = world.tube_params['exit']
		tube_width = world.tube_params['width']
		
		# Check if position is between entrance and exit y-coordinates
		in_y_range = pos[1] <= tube_entrance[1] and pos[1] >= tube_exit[1]
		
		# Check if position is within tube width
		in_x_range = abs(pos[0] - tube_entrance[0]) <= tube_width/2
		
		return in_y_range and in_x_range

	def get_agent_phase(self, agent: Agent, world: World):
		"""Helper function to determine agent's current phase"""
		pos = agent.state.p_pos
		in_tube = self.is_in_tube(world, pos)
		passed_tube = pos[1] < world.tube_params['exit'][1]
		
		if not in_tube and not passed_tube:
			return 0
		elif in_tube:
			return 1
		else:
			return 2


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
				if self.landmark_poses_occupied[goal] == self.coordination_range:
					goal_proximity = np.array([np.linalg.norm(self.landmark_poses[goal] - agent.state.p_pos)  for agent in world.agents])
					if not np.any(goal_proximity < self.min_dist_thresh):
						self.landmark_poses_occupied[goal] = np.min(goal_proximity)  # Reset falsely occupied goals

			if min_dist < self.min_dist_thresh:
				if agent.status == True:
					self.landmark_poses_occupied[top_two_indices[0]] = self.coordination_range
				else:
					self.landmark_poses_occupied[top_two_indices[0]] = self.coordination_range-min_dist
				self.goal_history[top_two_indices[0]] = agent.id

			else:
				# goal_proximity is finding how many agents are near this chosen goal
				goal_proximity = np.array([np.linalg.norm(agents_goal - agent.state.p_pos)  for agent in world.agents])
				closest_dist_to_goal = np.min(goal_proximity)


				# agent veered off the goal
				if self.landmark_poses_occupied[chosen_goal] == self.coordination_range:

					# if there are no agents on the goal, then the agent can take the goal and change the occupancy value
					if np.any(goal_proximity < self.min_dist_thresh):

						######
						## Add case when all nearby observed goals are occupied
						unoccupied_goals = self.landmark_poses[self.landmark_poses_occupied!= 1]
						unoccupied_goals_indices = np.where(self.landmark_poses_occupied != 1)[0]
						assert len(unoccupied_goals) > 0, f"All goals are occupied {self.landmark_poses_occupied}, {self.goal_history},{world.dists} {goal_proximity}"
						chosen_goal = np.argmin(np.linalg.norm(agent.state.p_pos - unoccupied_goals, axis=1))
						agents_goal = unoccupied_goals[chosen_goal]

					else:
						## add assertion to see if no goal is falsely occupied
						assert not np.any(goal_proximity < self.min_dist_thresh), f"Agent {agent.id} is not at goal {chosen_goal} but flag is set to occupied"
						self.landmark_poses_occupied[chosen_goal] = self.coordination_range-closest_dist_to_goal

				# another agent already at goal, can't overwrite the flag
				elif self.landmark_poses_occupied[chosen_goal] != self.coordination_range:
					self.landmark_poses_occupied[chosen_goal] = self.coordination_range-closest_dist_to_goal

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
		
		goal_pos, closest_goal_occupied, rel_second_closest_goal, goal_history = self.get_agent_nearby_goals(agent, world)
		goal_pos = self.get_rotated_position_from_relative(goal_pos, agent_heading)
		rel_second_closest_goal = self.get_rotated_position_from_relative(rel_second_closest_goal, agent_heading)
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

		# Rotate nearest neighbors relative to agent's heading
		rotated_neighbors = [
			self.get_rotated_position_from_relative(neighbor, agent_heading)
			for neighbor in nearest_neighbors
		]

		# Flatten into a single array
		rotated_neighbors = np.concatenate(rotated_neighbors)

		# Tube parameters
		tube_entrance = world.tube_params['entrance']
		tube_exit = world.tube_params['exit']
		tube_width = world.tube_params['width']
		
		# Calculate distances and directions to tube entrance/exit
		rel_to_entrance = tube_entrance - agent_pos
		rel_to_exit = tube_exit - agent_pos

		rot_rel_entrance = self.get_rotated_position_from_relative(rel_to_entrance, agent_heading)
		rot_rel_exit = self.get_rotated_position_from_relative(rel_to_exit, agent_heading)

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
			rot_rel_entrance,  # Vector to tube entrance
			rot_rel_exit,     # Vector to tube exit
			[tube_width],    # Tube width
			[phase]          # Current phase
		])

		return np.concatenate([
			[agent_speed],
			goal_pos,
			closest_goal_occupied,
			rel_second_closest_goal,
			rotated_neighbors,
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

	def graph_observation(self, agent:Agent, world:World) -> Tuple[arr, arr]:
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
		rel_pos = self.get_rotated_position_from_relative(rel_pos, agent.state.theta)
		rel_speed = entity_speed - agent_speed
		if 'agent' in entity.name:
			world.dists = np.array([np.linalg.norm(entity.state.p_pos - l) for l in self.landmark_poses])
			min_dist = np.min(world.dists)
			if min_dist < self.coordination_range:
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
			rel_goal_pos = self.get_rotated_position_from_relative(rel_goal_pos, agent.state.theta)
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
			return np.hstack([[rel_speed], rel_pos, rel_goal_pos,goal_occupied,goal_history,wall_o_corner,wall_d_corner,entity_type])

		else:
			raise ValueError(f'{entity.name} not supported')
		return np.hstack([[rel_speed], rel_pos, rel_goal_pos,goal_occupied,entity_type])