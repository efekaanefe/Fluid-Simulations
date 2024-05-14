if __name__=="__main__":

	import numpy as np
	import matplotlib.pyplot as plt


	# hyper-parameters
	Nx = 400
	Ny = 100
	Nt = 3000
	tau = 0.5

	num_directions = 9 # D2Q9
	discrete_velocities = np.array([[-1,1],[0,1],[1,1],
									[-1,0],[0,0],[1,0],
									[-1,-1],[0,-1],[1,-1]])
	weights = np.array([1/36,1/9,1/36,
						1/9, 4/9,1/9,
						1/36,1/9,1/36])

	
	# initialize the field 
	F = np.ones((Ny, Nx, num_directions)) + 0.01 * np.random.randn(Ny, Nx, num_directions)
	F[:,:, 5] = 3 # makes the flow field move to the right 

	obstacle_field = np.ones((Ny, Nx)) #  0 -> obstacle and 1 -> empty space
	
	def create_circular_obstacle(obstacle_field, center_position = None, radius = None):
		Ny, Nx = obstacle_field.shape

		if center_position is None: # use the middle of the image
			center_position = (int(Nx/2), int(Ny/2))
		if radius is None: # use the smallest distance between the center and image walls
			radius = min(center_position[0], center_position[1], Nx-center_position[0], Ny-center_position[1])

		cx, cy = center_position
		r = radius
		Y, X = np.ogrid[:Ny, :Nx]
		distance_from_center = np.sqrt((X - cx)**2 + (Y-cy)**2)
		mask = distance_from_center <= radius
		return mask

	mask = create_circular_obstacle(obstacle_field, (Nx * 0.8 ,Ny/2), 50)
	obstacle_field[~mask] = 0
	# plt.imshow(obstacle_field)
	# plt.show()

	# main loop
	for t_i in range(Nt): # time

		for direction in range(num_directions):
			for x_vel, y_vel in discrete_velocities[direction]:
				F[:,:,direction] =  np.roll(F[:,:,direction], x_vel, axis=1)
				F[:,:,direction] =  np.roll(F[:,:,direction], y_vel, axis=0)
				








