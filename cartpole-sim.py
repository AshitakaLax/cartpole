# CLASS: CS5890
# AUTHOR: Levi Balling
# DATE: Oct 20, 2018
# ASSIGNMENT: Cart-pole

#Imports
import math
import gym
# libraries for the re-inforced learning
import tensorflow as tf
import tflearn
import numpy as np

env = gym.make("CartPole-v0")
# constants
# +- 10 Newtons
FORCE_MAG = 10.0
LENGTH = 0.5
CART_MASS = 1.0
POLE_MASS = 0.1

# FORCE_MAG = 10.0
# LENGTH = 0.326
# CART_MASS = 0.711
# POLE_MASS = 0.209
# pole mass * Length of the pole
POLEMASS_LENGTH = POLE_MASS * LENGTH

TOTAL_MASS = CART_MASS + POLE_MASS
GRAVITY = 9.8
FOUR_THIRDS = 4.0/3.0
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
	return x, x_dot, theta, theta_dot

# Check whether the step should be failed
# this is deteriming what the reward should be
def CheckIfFailed(x, x_dot, theta, theta_dot):
	#calculate the theta Threashold
	theta_threshold = 12 * 2 * math.pi / 360
	x_threshold = 2.4
	return bool(x < -x_threshold or x > x_threshold or theta < -theta_threshold or theta > theta_threshold)


gamma = 0.99

def reward_update(rewards):
	temp_reward = 0
	result = np.zeros_like(rewards)
	for i in reversed(range(len(rewards))):
		result[i] = rewards[i] + gamma * temp_reward
		temp_reward += rewards[i]
	return result



# the shape has to be the format of the 4 parameters after action
input_net = tflearn.input_data(shape=[None, 4])
episode_net = tflearn.fully_connected(input_net, 256, activation="relu")
episode_net = tflearn.fully_connected(episode_net, 256, activation="relu")
episode_net = tflearn.fully_connected(episode_net, 256, activation="relu")
result_net = tflearn.fully_connected(episode_net, 2, activation="softmax")

rewards = tf.placeholder(tf.float32, [None])
actions = tf.placeholder(tf.int32, [None])
outputs = tf.gather(tf.reshape(result_net, [-1]), tf.range(0, tf.shape(result_net)[0] * tf.shape(result_net)[1], 2) + actions)
loss = -tf.reduce_mean(tf.log(outputs) * rewards)
optimizer = tf.train.AdamOptimizer()
update = optimizer.minimize(loss)

# Setup Episodes to run the simulation
Episodes = 20000
Max_iterations = 100

# The 
reward_sum_arr = []
data_set = []

model_saver = tf.train.Saver()
with tf.Session() as episode_session:

	episode_session.run(tf.global_variables_initializer())
	for i in range(Episodes):
		#obs = env.reset()
		#print("Episode: " + str(i))
		epi_reward = 0
		epi_history = []
		cart_pole_params = np.random.uniform(low=-0.05, high=0.05, size=(4,))
		for j in range(Max_iterations):
			iteration_result = episode_session.run(result_net, feed_dict={input_net: [cart_pole_params]}).reshape(2)
			#iteration_result = episode_session.run(result_net, feed_dict={input_net: [obs]}).reshape(2)
			action = np.random.choice(iteration_result, p=iteration_result)
			action = np.argmax(iteration_result == action)
			#obs1, r, d, _ = env.step(action)
			#reward = r
			#failed = d
			x, x_dot, theta, theta_dot = cart_pole(action, cart_pole_params[0], cart_pole_params[1], cart_pole_params[2], cart_pole_params[3])
			failed = CheckIfFailed(x, x_dot, theta, theta_dot)
			reward = 0.0
			if not failed:
				reward = 1.0
			
			epi_reward += reward
			#epi_history.append([[obs[0], obs[1], obs[2], obs[3]], reward, action])
			epi_history.append([cart_pole_params[0], cart_pole_params[1], cart_pole_params[2], cart_pole_params[3], reward, action])
			#obs = obs1
			cart_pole_params[0] = x
			cart_pole_params[1] = x_dot
			cart_pole_params[2] = theta
			cart_pole_params[3] = theta_dot
			
			# end of the episode
			if failed:
				# failed means that the ended the last round, but we still want to have
				# a positive reward
				reward_sum_arr.append(epi_reward + 1.0)
				epi_history = np.array(epi_history)
				#epi_history[:, 1] = reward_update(epi_history[:, 1])
				epi_history[:, 4] = reward_update(epi_history[:, 4])
				data_set.extend(epi_history)
				if i % 10 == 0 and i != 0:
					data_set = np.array(data_set)
					shaped_data = data_set[:,[0,1,2,3]]
					#shaped_data = np.vstack(data_set[:, 0])
					#episode_session.run(update, feed_dict={input_net: shaped_data, rewards: data_set[:, 1], actions: data_set[:, 2]})
					episode_session.run(update, feed_dict={input_net: shaped_data, rewards: data_set[:, 4], actions: data_set[:, 5]})
					data_set = []
				break
		if i % 50 == 0 and i != 0:
			print(np.mean(reward_sum_arr[-50:]))
			if np.mean(reward_sum_arr[-50:]) == 100:
				break
	model_saver.save(episode_session, "models/cart_pole.ckpt")

avg_reward = [np.mean(reward_sum_arr[i-10:i+10]) for i in range(10, len(reward_sum_arr))]
print(avg_reward[::10])





			












