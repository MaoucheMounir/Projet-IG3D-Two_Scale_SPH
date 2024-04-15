# -*- coding: utf-8 -*-
"""
Created on Mar 30 2024

@author: MAOUCHE
"""

from mpl_toolkits import mplot3d
import numpy as np
import matplotlib.pyplot as plt
from scipy.special import gamma

"""
Simulation of the structure of a star using two-scale SPH.
Based on code by P. Mocz and a paper by Solenthaler & Gross, 2011
"""

def W( x, y, z, h ):
	"""
    Gausssian Smoothing kernel (3D)
	x     is a vector/matrix of x positions
	y     is a vector/matrix of y positions
	z     is a vector/matrix of z positions
	h     is the smoothing length
	w     is the evaluated smoothing function
	"""
	
	r = np.sqrt(x**2 + y**2 + z**2)
	
	w = (1.0 / (h*np.sqrt(np.pi)))**3 * np.exp( -r**2 / h**2)
	
	return w
	
	
def gradW( x, y, z, h ):
	"""
	Gradient of the Gausssian Smoothing kernel (3D)
	x     is a vector/matrix of x positions
	y     is a vector/matrix of y positions
	z     is a vector/matrix of z positions
	h     is the smoothing length
	wx, wy, wz     is the evaluated gradient
	"""
	
	r = np.sqrt(x**2 + y**2 + z**2)
	
	n = -2 * np.exp( -r**2 / h**2) / h**5 / (np.pi)**(3/2)
	wx = n * x
	wy = n * y
	wz = n * z
	
	return wx, wy, wz
	
	
def getPairwiseSeparations( ri, rj ):
	"""
	Get pairwise desprations between 2 sets of coordinates
	ri    is an M x 3 matrix of positions
	rj    is an N x 3 matrix of positions
	dx, dy, dz   are M x N matrices of separations
	"""
	
	M = ri.shape[0]
	N = rj.shape[0]
	
	# positions ri = (x,y,z)
	rix = ri[:,0].reshape((M,1))
	riy = ri[:,1].reshape((M,1))
	riz = ri[:,2].reshape((M,1))
	
	# other set of points positions rj = (x,y,z)
	rjx = rj[:,0].reshape((N,1))
	rjy = rj[:,1].reshape((N,1))
	rjz = rj[:,2].reshape((N,1))
	
	# matrices that store all pairwise particle separations: r_i - r_j
	dx = rix - rjx.T
	dy = riy - rjy.T
	dz = riz - rjz.T
	
	return dx, dy, dz
	

def getDensity( r, pos, m, h ):
	"""
	Get Density at sampling loctions from SPH particle distribution
	r     is an M x 3 matrix of sampling locations
	pos   is an N x 3 matrix of SPH particle positions
	m     is the particle mass
	h     is the smoothing length
	rho   is M x 1 vector of densities
	"""
	
	M = r.shape[0]
	
	dx, dy, dz = getPairwiseSeparations( r, pos );
	
	rho = np.sum( m * W(dx, dy, dz, h), 1 ).reshape((M,1))
	
	return rho
	
	
def getPressure(rho, k, n):
	"""
	Equation of State
	rho   vector of densities
	k     equation of state constant
	n     polytropic index
	P     pressure
	"""
	
	P = k * rho**(1+1/n)
	
	return P
	

def getAcc( pos, vel, m, h, k, n, lmbda, nu ):
	"""
	Calculate the acceleration on each SPH particle
	pos   is an N x 3 matrix of positions
	vel   is an N x 3 matrix of velocities
	m     is the particle mass
	h     is the smoothing length
	k     equation of state constant
	n     polytropic index
	lmbda external force constant
	nu    viscosity
	a     is N x 3 matrix of accelerations
	"""
	
	N = pos.shape[0]
	
	# Calculate densities at the position of the particles
	rho = getDensity( pos, pos, m, h )
	
	# Get the pressures
	P = getPressure(rho, k, n)
	
	# Get pairwise distances and gradients
	dx, dy, dz = getPairwiseSeparations( pos, pos )
	dWx, dWy, dWz = gradW( dx, dy, dz, h )
	
	# Add Pressure contribution to accelerations
	ax = - np.sum( m * ( P/rho**2 + P.T/rho.T**2  ) * dWx, 1).reshape((N,1))
	ay = - np.sum( m * ( P/rho**2 + P.T/rho.T**2  ) * dWy, 1).reshape((N,1))
	az = - np.sum( m * ( P/rho**2 + P.T/rho.T**2  ) * dWz, 1).reshape((N,1))
	
	# pack together the acceleration components
	a = np.hstack((ax,ay,az))
	
	# Add external potential force
	a -= lmbda * pos
	
	# Add viscosity
	a -= nu * vel
	
	return a
	


def main():
	""" SPH simulation """
	
	# Simulation parameters
	N         = 800    # Number of particles
	t         = 0      # current time of the simulation
	tEnd      = 4     # time at which simulation ends
	dt        = 0.01   # timestep
	M         = 2      # star mass
	R         = 0.75   # star radius
	h         = 0.1    # smoothing length
	k         = 0.1    # equation of state constant
	n         = 1      # polytropic index
	nu        = 1      # damping
	plotRealTime = True # switch on for plotting as the simulation goes along
	
	# Generate Initial Conditions
	np.random.seed(42)            # set the random number generator seed
	
	lmbda = 2*k*(1+n)*np.pi**(-3/(2*n)) * (M*gamma(5/2+n)/R**3/gamma(1+n))**(1/n) / R**2  # ~ 2.01
	m     = M/N                    # single particle mass
	pos   = np.random.randn(N,3)   # randomly selected positions and velocities
	vel   = np.zeros(pos.shape)
	
	# calculate initial gravitational accelerations
	acc = getAcc( pos, vel, m, h, k, n, lmbda, nu )
	
	# number of timesteps
	Nt = int(np.ceil(tEnd/dt))
	
	# prep figure
	fig = plt.figure(figsize=(6.4,3.8), dpi=100)
	ax = plt.axes(projection ="3d")
    
	# rendring window : circle centred on (x0,y0) with radius r0
	x0 = 0 #x0 = 0.2    
	y0 = 0 #y0 = 0.2  
	z0 = 0 #z0 = 0.2         
	r0 = 0.4    	
    
	# Simulation Main Loop
	for i in range(Nt):
		# (1/2) kick
		vel += acc * dt/2
		
		# drift
		pos += vel * dt
		
		# update accelerations
		acc = getAcc( pos, vel, m, h, k, n, lmbda, nu )
		
		# (1/2) kick
		vel += acc * dt/2
		
		# update time
		t += dt
		
		# get density for plotting
		rho = getDensity( pos, pos, m, h )

		# computing distance of each particle from center of the rendring cercle
		posr = ( ( pos[:,0] - x0 )**2  +  ( pos[:,1] - y0 )**2 +  ( pos[:,2] - z0 )**2 )**0.5

		# adding column posr to pos
		pose = np.concatenate((pos,posr.reshape((pos.shape[0],1))),axis=1)

		# prepare rendering matrix
		poseH = np.zeros([N,4])  #High resolution for particles within the window
		poseL = np.zeros([N,4])  #Low resolution for particles out of the window 
		poseHL = np.zeros([N,4]) #concatenation of poseH and poseL
		ih = -1  # row index of poseH
		il = -1  # row index of poseL
        
        ########################################################################
        ######### Number of particles rendered in Low resolution ################
		nbL = 2  # 2 means only 1 of 2 particles, out of the window, are rendred
        ########################################################################
        ########################################################################
		
		reducesize = 0
		for i in range(N) :
			if (pose[i,3] <= r0) :
				ih += 1
				poseH[ih,:] = pose[i,:]        
			else :
				reducesize += 1
				if (reducesize == nbL) :
					il += 1
					poseL[il,:] = pose[i,:]
					reducesize = 0

		poseHL = poseH
		poseHL[ih+1:ih+il+2,:]=poseL[0:il+1,:]

		poseHL = np.delete(poseHL, range(il+ih+2, N), 0) 
		posHL = poseHL[:,0:3]

		# get density for plotting HL
		rhoHL = getDensity( posHL, posHL, m, h )


		# plot in real time
		if plotRealTime or (i == Nt-1):
			plt.sca(ax)
			plt.cla()
			cval = np.minimum((rhoHL-3)/3,1).flatten()
			ax.scatter3D(posHL[:,0],posHL[:,1],posHL[:,2], c=cval, cmap=plt.cm.autumn, s=10, alpha=0.5)
			
			ax.set(xlim=(-1.4, 1.4), ylim=(-1.2, 1.2),zlim=(-1.4, 1.4))
			ax.set_aspect('equal', 'box')
			ax.set_xticks([-1,0,1])
			ax.set_yticks([-1,0,1])
			ax.set_zticks([-1,0,1])
			ax.set_facecolor('black')
			ax.set_facecolor((.1,.1,.1))
			
			plt.pause(0.00001)
	    
	
	
	# Save figure
	plt.savefig('sph.png',dpi=240)
	plt.show()
	    
	return 0
	


  
if __name__== "__main__":
  main()