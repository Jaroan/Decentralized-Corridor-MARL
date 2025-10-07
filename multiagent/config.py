import numpy as np


class AirTaxiConfig():
    V_MIN = 60 * 0.514444 * 0.001 # knots to km/s
    V_MAX = 175 * 0.514444  * 0.001 # knots to km/s
    V_NOMINAL = 110 * 0.514444  * 0.001 # knots to km/s
    ACCEL_MIN = -0.001 # km/s^2
    ACCEL_MAX = 0.002 # km/s^2
    ANGULAR_RATE_MAX = 0.1 # rad/s
    MOTION_PRIM_ACCEL_OPTIONS = 5
    MOTION_PRIM_ANGRATE_OPTIONS = 5
    CBF_RATE = 3.0

    ENGAGEMENT_DISTANCE = 1.4
    ENGAGEMENT_DISTANCE_REFERENCE_SEPARATION_DISTANCE = 2200 * 0.0003048
    
    DT = 1.0
    # DISTANCE_TO_GOAL_THRESHOLD = 750 * 0.0003048 # ft to km
    DISTANCE_TO_GOAL_THRESHOLD = 0.35
    GOAL_HEADING_THRESHOLD = np.pi/4
    # GOAL_SPEED_THRESHOLD = 20 * 0.514444 * 0.001 # knots to km/s
    GOAL_SPEED_THRESHOLD = 0.03 # knots to km/s
    
    # Ref: Preliminary Analysis of Separation Standards for Urban Air Mobility Using Unmitigated Fast-Time
    # Test params: 1500, 1800, 2200, 5000 ft
    SEPARATION_DISTANCE = 1500 * 0.0003048 # (first parameter in ft, converted to km)
    # COORDINATION_RANGE = 5 # 3 miles to km
    COORDINATION_RANGE = 3 * 1.60934 # 3 miles to km
    VALUE_FUNCTION_FILE_NAME = 'data/airtaxi_value_function.pkl'
    TTR_FILE_NAME = 'data/airtaxi_ttr_function.pkl'


class UnicycleVehicleConfig():
    V_MIN = 0.4
    V_MAX = 0.75
    V_NOMINAL = 0.5
    ACCEL_MIN = -0.5
    ACCEL_MAX = 0.5
    ANGULAR_RATE_MAX = 0.5
    MOTION_PRIM_ACCEL_OPTIONS = 5 ## choices are 3 and 5
    MOTION_PRIM_ANGRATE_OPTIONS = 5 ## total motion primitive choices are MOTION_PRIM_ACCEL_OPTIONS * MOTION_PRIM_ANGRATE_OPTIONS

    # simulation timestep
    DT = 0.1
    # agent within this distance to the landmark is considered to have reached the goal.
    DISTANCE_TO_GOAL_THRESHOLD = 0.2
    # separation distance between agents for safety
    COLLISION_DISTANCE = 0.4
    # communication distance (entities within this distance are considered in each agent's observations)
    COMMUNICATION_RANGE = 5

# class UnicycleVehicleConfig():
#     V_MIN = 0.04 # km/s scaled by 10
#     V_MAX = 0.075 #  km/s
#     V_NOMINAL = 0.6 #  km/s
#     ACCEL_MIN = -0.005 # km/s^2
#     ACCEL_MAX = 0.005 # km/s^2
#     ANGULAR_RATE_MAX = 0.5# rad/s
#     MOTION_PRIM_ACCEL_OPTIONS = 5 ## choices are 3 and 5
#     MOTION_PRIM_ANGRATE_OPTIONS = 5 ## total motion primitive choices are MOTION_PRIM_ACCEL_OPTIONS * MOTION_PRIM_ANGRATE_OPTIONS

#     # simulation timestep
#     DT = 1
#     # agent within this distance to the landmark is considered to have reached the goal.
#     DISTANCE_TO_GOAL_THRESHOLD = 0.2 #km
#     # separation distance between agents for safety
#     COLLISION_DISTANCE = 0.4 # km
#     # communication distance (entities within this distance are considered in each agent's observations)
#     COMMUNICATION_RANGE = 5

# class UnicycleVehicleConfig():
#     V_MIN = 0.04 # km/s scaled by 10
#     V_MAX = 0.075 #  km/s
#     V_NOMINAL = 0.6 #  km/s
#     ACCEL_MIN = -0.005 # km/s^2
#     ACCEL_MAX = 0.005 # km/s^2
#     ANGULAR_RATE_MAX = 0.5# rad/s
#     MOTION_PRIM_ACCEL_OPTIONS = 5 ## choices are 3 and 5
#     MOTION_PRIM_ANGRATE_OPTIONS = 5 ## total motion primitive choices are MOTION_PRIM_ACCEL_OPTIONS * MOTION_PRIM_ANGRATE_OPTIONS

#     # simulation timestep
#     DT = 1
#     # agent within this distance to the landmark is considered to have reached the goal.
#     DISTANCE_TO_GOAL_THRESHOLD = 0.2 #km
#     # separation distance between agents for safety
#     COLLISION_DISTANCE = 0.4 # km
#     # communication distance (entities within this distance are considered in each agent's observations)
#     COMMUNICATION_RANGE = 5


class DoubleIntegratorConfig():
    VX_MIN = -1.0
    VX_MAX = 1.0
    VY_MIN = -1.0  
    VY_MAX = 1.0
    # Only used for goal point target speed.
    V_MIN = 0.1
    V_NOMINAL = 0.5
    V_MAX = np.sqrt(VX_MAX**2 + VY_MAX**2)
    ACCELX_MIN = -1.0
    ACCELX_MAX = 1.0
    ACCELY_MIN = -1.0
    ACCELY_MAX = 1.0
    ACCELX_OPTIONS = 3 # double check appropriate value
    ACCELY_OPTIONS = 3 # double check appropriate value
    DT = 0.1
    DISTANCE_TO_GOAL_THRESHOLD = 0.2 # m
    # separation distance between agents for safety
    COLLISION_DISTANCE = 0.5
    # communication distance (entities within this distance are considered in each agent's observations)
    COMMUNICATION_RANGE = 5
    # Dummy heading threshold (not relevant for holonomic agents)
    GOAL_HEADING_THRESHOLD = np.pi


class RewardWeightConfig():
    # min and max reward at each timestep.
    MIN_REWARD = -40
    MAX_REWARD = 50
    
    GOAL_REACH = 50
    CONFLICT = -20 # if agent is within separation distance of another agent

eval_scenario_type = "left_to_right_merge_and_land"