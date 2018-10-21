# CLASS: CS5890
# AUTHOR: Levi Balling
# DATE: Oct 20, 2018
# ASSIGNMENT: Cart-pole

#Imports
import math

# constants
# +- 10 Newtons
FORCE_MAG = 10.0
POLEMASS_LENGTH = 0.326

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
	temp = (force + )


