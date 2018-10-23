# CLASS: CS5890
# AUTHOR: Levi Balling
# DATE: Oct 20, 2018
# ASSIGNMENT: Cart-pole

#Imports
import math
# libraries for the re-inforced learning
import tensorflow as tf
import tflearn
import numpy as np

# constants
# +- 10 Newtons
FORCE_MAG = 10.0

LENGTH = 0.326
CART_MASS = 0.711
POLE_MASS = 0.209
# pole mass * Length of the pole
POLEMASS_LENGTH = POLE_MASS * LENGTH

TOTAL_MASS = CART_MASS + POLE_MASS
GRAVITY = 9.82
FOUR_THIRDS = 4/3
TAU = 0.02

# this is the equivalent of gym cartpole.py:step(self, action)
def cart_pole(action, x, x_dot, theta, theta_dot):
	"""runs the simulation of the cart pole
	
	Arguments:
		action {int} -- The action to take
		x {[type]} -- [description]
		x_dot {[type]} -- [description]
		theta {[type]} -- [description]
		theta_dot {[type]} -- [description]
	"""
	x_acc = 0.0
	theta_acc = 0.0
	force = 0.0
	cos_theta = 0.0
	sin_theta = 0.0
	temp = 0.0

	force = FORCE_MAG if (action > 0)  else -FORCE_MAG
	cos_theta = math.cos(theta)
	sin_theta = math.sin(theta)
	temp = (force + POLEMASS_LENGTH * theta_dot * theta_dot * sin_theta) / TOTAL_MASS
	
	# calculate Theta acceleration
	theta_acc = (GRAVITY * sin_theta - cos_theta * temp) / (LENGTH * (FOUR_THIRDS - POLE_MASS * cos_theta * cos_theta / TOTAL_MASS))

	x_acc = temp - POLEMASS_LENGTH * theta_acc * cos_theta / TOTAL_MASS

	x = x + (TAU * x_dot)
	x_dot = x_dot + (TAU * x_acc)
	theta = theta + (TAU * theta_dot)
	theta_dot = theta_dot + (TAU * theta_acc)

# Check whether the step should be failed
def CheckIfFailed(x, x_dot, theta, theta_dot):
	#calculate the theta Threashold
	theta_threshold = 12 * 2 * math.pi / 360
	x_threshold = 2.4
	return bool(x < -x_threshold or x > x_threshold or theta < -theta_threshold or theta > theta_threshold)



# Setup Episodes to run the simulation
Episodes = 10
Max_iterations = 5000

# The 
s_t = []

with tf.Session() as episode_session:
	for i in range(Episodes):
		print("Episode: " + str(i))
		# the shape has to be the format of the 4 parameters after action
		episode_net = tflearn.input_data(shape=[None, 4])
		episode_net = tflearn.fully_connected(episode_net, 256, activation="relu")
		episode_net = tflearn.fully_connected(episode_net, 256, activation="relu")
		episode_net = tflearn.fully_connected(episode_net, 256, activation="relu")
		result_net = tflearn.fully_connected(episode_net, 2, activation="softmax")

		rewards = tf.placeholder(tf.float32, [None])
		actions = tf.placeholder(tf.int32, [None])
		outputs = tf.gather(tf.reshape(result_net, [-1]), tf.range(0, tf.shape(result_net)[0] * tf.shape(result_net)[1], 2) + actions)
		loss = -tf.reduce_mean(tf.log(outputs) * rewards)
		optimizer = tf.train.AdamOptimizer()
		update = optimizer.minimize(loss)











