if __name__=="__main__":

	import numpy as np
	import matplotlib.pyplot as plt


	# hyper-parameters
	Nx = 400
	Ny = 100
	Nt = 10000
	tau = 0.53

	num_directions = 9 # D2Q9
	discrete_directions = np.array([[-1,1],[0,1],[1,1],  # for velocities
									[-1,0],[0,0],[1,0],
									[-1,-1],[0,-1],[1,-1]])
	weights = np.array([1/36,1/9,1/36,
						1/9, 4/9,1/9,
						1/36,1/9,1/36])
	
	# initialize the field 
	F = np.ones((Ny, Nx, num_directions)) + 0.01 * np.random.randn(Ny, Nx, num_directions)
	F[:,:, 5] = 2.3 # makes the flow field move to the right 

	obstacle_field = np.zeros((Ny, Nx)) #  0 -> obstacle and 1 -> empty space
	
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
		mask = distance_from_center >= radius
		return mask

	mask = create_circular_obstacle(obstacle_field, (Nx//4, Ny//2), 13)
	obstacle_field[~mask] = 1
	obstacle_field = np.array(obstacle_field, dtype=bool)


	# main loop
	for t_i in range(Nt): # time
		
		
		# absorb BC
		F[:,-1,[0,3,6]] = F[:,-2,[0,3,6]] 
		F[:,0,[2,5,8]] = F[:,0,[2,5,8]] 


		# advect velocities
		for direction in range(num_directions):
			x_dir, y_dir = discrete_directions[direction]
			F[:,:,direction] =  np.roll(F[:,:,direction], x_dir, axis=1)
			F[:,:,direction] =  np.roll(F[:,:,direction], y_dir, axis=0)

		# apply reaction forces to make no slip BC
		boundary_F = F[obstacle_field, :]
		boundary_F = boundary_F[:, [8, 7, 6, 5, 4, 3, 2, 1, 0]] # these are the opposite direction of each node (vel_dir)

		# calculating fluid variables
		rho = np.sum(F, axis=2)
		ux = np.sum(F * discrete_directions[:,0], axis=2)/rho # x momentum
		uy = np.sum(F * discrete_directions[:,1], axis=2)/rho # y momentum

		F[obstacle_field, :] = boundary_F
		ux[obstacle_field] = 0 
		uy[obstacle_field] = 0

		# collisions
		F_eq = np.zeros_like(F)
		for direction in range(num_directions):
			x_dir, y_dir = discrete_directions[direction]; w = weights[direction]

			F_eq[:,:,direction] = rho * w * (
						1 + 3*(x_dir*ux + y_dir*uy) + 9*(x_dir*ux + y_dir*uy)**2 /2 - 3*(ux**2 + uy**2)/2 )
			
		F = F + -1/tau * (F - F_eq)

		if t_i%100 == 0:
			## plot curl
			dfydx = ux[2:, 1:-1] - ux[0:-2,1:-1]
			dfxdy = uy[1:-1,2:] - uy[1:-1, 0:-2]
			curl = dfydx - dfxdy
			plt.imshow(curl, cmap="bwr")
			plt.pause(0.01)
			plt.cla()

			## plot velocities
			# plt.imshow(np.sqrt(ux**2 + uy**2))
			# plt.pause(0.01)
			# plt.cla()

			


		

				








