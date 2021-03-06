#** 
# @author      Skye Leake <skleake96@gmail.com>
# @version     0.1.0
# @since       0.0.0
#/

#**
# Composes transformation matrix from input scale, rotation, shear and translation parameters
# @param  scale 				- [float] (inf, inf) zero colapses all to a point, negative values reflect
# @param  rotaion 				- [float] [-pi/2, pi/2] rotation about (0,0,0) as measured from x+ (right handed) 
# @param  shear 				- [float] shear along a primary axis (not implemented)
# @param  translation 			- [float] translation along a primary axis
# @return Transformation matrix - [float] calcualted: scale by rotation_y by rotation_x by rotation_z by translation
#/

# --- Imports ---
import numpy as np 																		  # ver. 1.15.0

def comp_matrix(scale, rotation, shear, translation):
	# should filter inputs if accepting error prone input (dtype, length, and domains)
	Tx = translation[0]
	Ty = translation[1]																	  # eff: we only ever use these variables once, no need to call them, parse input as needed
	Tz = 0 if translation.size < 3 else translation[2]
	Sx = scale[0]
	Sy = scale[1]
	Sz = 1 if scale.size < 3 else scale[2]
	Shx = shear[0]
	Shy = shear[1]
	Shz = 0 if shear.size < 3 else shear[2]
	Rxc, Rxs = np.cos(rotation[0]), np.sin(rotation[0])
	Ryc, Rys = np.cos(rotation[1]), np.sin(rotation[1])									  # eff: we call these variables multiple times, create standalones for efficency
	Rzc, Rzs = (1,0) if rotation.size < 3 else (np.cos(rotation[2]), np.sin(rotation[2]))

	T_M = np.array([[1, 0, 0, Tx],
                    [0, 1, 0, Ty],
                    [0, 0, 1, Tz],
                    [0, 0, 0, 1]])
	S_M = np.array([[Sx, 0, 0, 0],
                    [0, Sy, 0, 0],
                    [0, 0, Sz, 0],
                    [0, 0, 0, 1]])
	Sh_M = np.array([[1, Shy/Shx, Shz/Shx, 0],
                     [Shx/Shy, 1, Shz/Shy, 0],
                     [Shx/Shz, Shy/Shz, 1, 0],
                     [0, 0, 0, 1]])
	Rx_M = np.array([[1, 0, 0, 0],
                     [0, Rxc, -Rxs, 0],
                     [0, Rxs, Rxc, 0],
                     [0, 0, 0, 1]])
	Ry_M = np.array([[Ryc, 0, Rys, 0],
                     [0, 1, 0, 0],
                     [-Rys, 0, Ryc, 0],
                     [0, 0, 0, 1]])
	Rz_M = np.array([[Rzc, -Rzs, 0, 0],
                     [Rzs, Rzc, 0, 0],
                     [0, 0, 1, 0],
                     [0, 0, 0, 1]])

	# TODO: Add support for off primary axis rotation (rotation about an arbitary axis)

	#return np.dot(T_M,np.dot(Rz_M,np.dot(Rx_M,np.dot(Ry_M,S_M))))						  # IMPORTANT: the transformations must be multiplied together in the [B]reverse order[/B] to that in which we want them applied
	return np.dot(S_M,np.dot(T_M, Rz_M))
#**
# Decompose transformation matrix into its component parts (Not used and probally full of bugs)
# @param  transformation_matrix - [floats] 4x4 transformation matrix 
# @return component_array - [[floats],[floats],[floats],[floats]] scale, rotation, shear, translation
#/
def decomp_matrix(transformation_matrix):
	tm = transformation_matrix
	translation = np.array([tm[0,3], tm[1,3], tm[2,3]])
	scale = np.array([np.abs(np.sqrt(np.power(tm[0,0],2)+np.power(tm[1,0],2)+np.power(tm[2,0],2))),
					  np.abs(np.sqrt(np.power(tm[0,1],2)+np.power(tm[1,1],2)+np.power(tm[2,1],2))),
					  np.abs(np.sqrt(np.power(tm[0,2],2)+np.power(tm[1,2],2)+np.power(tm[2,2],2)))])
	#rotation = np.array([np.arctan2(tm[2,1]/scale[1],tm[2,2],scale[2]),
	#					 np.arctan2(-tm[2,0]/scale[0],np.sqrt(np.power(tm[2,1]/scale[1],2)+np.power(tm[2,2]/scale[2],2))),
	#					 np.arctan2(tm[1,0]/scale[0],tm[0,0]/scale[0])])
	rotation = np.ones(3)
	shear = np.zeros(3)																	  # TODO: support for shear

	return 	np.array([scale, rotation, shear, translation])