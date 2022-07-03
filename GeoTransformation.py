import numpy as np
from skimage.color import *

# small number
eps = 1e-13

# since numpy uses round half to even (https://en.wikipedia.org/wiki/Rounding#Round_half_to_even) this function is used instead
def roundd(x):
    return (np.floor(x+0.5)).astype('int')

#######################################################################################
#                           Geometrical Transformation                                #
#                                                                                     #
#  (ul_o)  *------------* (ur_o)                                                      #
#          |            |                                                             #
#          |            |                                                             #
#          |            |                                                             #
#  (ll_o)  *------------* (lr_o)                                                      #
#                                                                                     #
#                                                                                     #
# the transformed corners are ul_t, ur_t, ...                                         #
#                                                                                     #
# input:                                                                              #
#       img            image to be transformed                                        #
#                                                                                     #
#       matrix         3x3 transformation matrix (numpy-array NOT numpy-matrix)       #
#                                                                                     #
#       method         method of interpolation (either nearest neighbour or bilinear) #
#                                                                                     #
#       check_runtime  measure runtime                                                #
#                                                                                     #
# returns:                                                                            #
#       transformed image                                                             #
#       displacement-vector                                                           #
#######################################################################################

def geoTransformation(img, matrix, method='nn'):
    data_type = img.dtype

    # do some validation
    if (method!='nn' and method!='bilinear'):
        raise ValueError('unknown interpolation method ...')

    if (matrix.shape != (3,3)):
        raise ValueError('matrix has wrong size ...')

    if (img.dtype !='uint8'):
        raise ValueError('currently only images of datatype uint8 are suppported ...')

    # if image is not a grayscale image make it one
    if (len(img.shape)!=2):
        img = (rgb2gray(img)*255).astype('uint8')

    # width and height of original image
    height_o, width_o = img.shape

    #########################################################
    # Inner function for calculating bilinear interpolation #
    #########################################################
    def bilinear_interpolation(x, y):

        # to ensure that all indices are valid
        x = np.where(eps<x,x,eps)
        x = np.where(x<height_o-1-eps,x,height_o-1-eps)
        y = np.where(eps<y,y,eps)
        y = np.where(y<width_o-1-eps,y,width_o-1-eps)

        # get row index i and column index j of upper left corner
        i = np.floor(x).astype('int')
        j = np.floor(y).astype('int')

        # get image values from all four neighbours
        I_0 = img[i,j]
        I_1 = img[i,j+1]
        I_2 = img[i+1,j]
        I_3 = img[i+1,j+1]

        result = (1-x+i)*(1-y+j)*I_0+(1-x+i)*(y-j)*I_1+(x-i)*(1-y+j)*I_2+(x-i)*(y-j)*I_3

        return (roundd(result)).astype('uint8')

    # inverse transformation necessary for interpolation
    invMatrix = np.linalg.inv(matrix)

    # determine homogeneous coordinates of transformed corners
    ul_o = np.array([0,0,1])
    ur_o = np.array([0,width_o-1,1])
    lr_o = np.array([height_o-1,width_o-1,1])
    ll_o = np.array([height_o-1,0,1])


    ul_t = np.matmul(matrix, ul_o)
    ur_t = np.matmul(matrix, ur_o)
    lr_t = np.matmul(matrix, lr_o)
    ll_t = np.matmul(matrix, ll_o)

    # dehomogenize coordinate vectors
    ul_t = 1.0/ul_t[2]*ul_t
    ur_t = 1.0/ur_t[2]*ur_t
    lr_t = 1.0/lr_t[2]*lr_t
    ll_t = 1.0/ll_t[2]*ll_t

    # create bounding box
    min_x = np.min([ul_t[0], ur_t[0], lr_t[0], ll_t[0]])
    min_y = np.min([ul_t[1], ur_t[1], lr_t[1], ll_t[1]])

    # store displacement vector
    displacement = np.array([[min_x], [min_y], [0]])

    # transformed image's size
    height_t = roundd(np.max([ul_t[0], ur_t[0], lr_t[0], ll_t[0]])-np.min([ul_t[0], ur_t[0], lr_t[0], ll_t[0]]))+1
    width_t  = roundd(np.max([ul_t[1], ur_t[1], lr_t[1], ll_t[1]])-np.min([ul_t[1], ur_t[1], lr_t[1], ll_t[1]]))+1

    # define constant gray value as default
    img_t = np.full((height_t,width_t),128).astype('uint8')

    ###################################
    # start interpolating gray values #

    # row- and column indices in new image as arrays
    r_new_idx = np.arange(height_t).astype('int')
    c_new_idx = np.arange(width_t).astype('int')

    # indices prepared for vectorising
    idx_new      = np.ones((3,len(r_new_idx)*len(c_new_idx)))
    idx_new[0,:] = np.tile(r_new_idx, len(c_new_idx))
    idx_new[1,:] = np.repeat(c_new_idx,len(r_new_idx))

    idx_old = np.matmul(invMatrix,(idx_new + np.tile(displacement,idx_new.shape[1])))

    if (method=='nn'):

        # use nearest neighbour interpolation
        # actual condition ensuring validity of indices in original image coordinates
        r_old_idx=roundd(idx_old[0,:]/idx_old[2,:])
        c_old_idx=roundd(idx_old[1,:]/idx_old[2,:])

        indices_o = np.array([r_old_idx, c_old_idx])
        indices_n = idx_new[0:2].astype('int')

        condition = np.where((indices_o[0]>=0) & (indices_o[0]<height_o) & (indices_o[1]>=0) & (indices_o[1]<width_o))
        indices_o = indices_o.T[condition].T
        indices_n = indices_n.T[condition].T

        img_t[indices_n[0], indices_n[1]]=img[indices_o[0],indices_o[1]]

    else:
        # use bilinear interpolation
        # actual condition ensuring validity of indices in original image coordinates
        r_old_idx=idx_old[0,:]/idx_old[2,:]
        c_old_idx=idx_old[1,:]/idx_old[2,:]

        indices_o = np.array([r_old_idx, c_old_idx])
        indices_n = idx_new[0:2].astype('int')

        condition_bl = np.where((indices_o[0]>=0) & (indices_o[0]<=height_o-1) & (indices_o[1]>=0) & (indices_o[1]<=width_o-1))
        indices_o_bl = indices_o.T[condition_bl].T
        indices_n_bl = indices_n.T[condition_bl].T

        # perform bilinear interpolation
        img_t[indices_n_bl[0], indices_n_bl[1]]=bilinear_interpolation(indices_o_bl[0],indices_o_bl[1])

        # due to unavoidable rounding errors there might be some relevant indices within a narrow gap around the original image
        # while these cannot be considered using bilinear interpolation we apply nearest neigbour interpolation for those
        condition_up_nn    = (indices_o[0]>-0.5) & (indices_o[0]<0) & (indices_o[1]>-0.5) & (indices_o[1]<width_o-1+0.5)
        condition_left_nn  = (indices_o[0]>-0.5) & (indices_o[0]<height_o-1+0.5) & (indices_o[1]>-0.5) & (indices_o[1]<0)
        condition_right_nn = (indices_o[0]>-0.5) & (indices_o[0]<height_o-1+0.5) & (indices_o[1]>width_o-1) & (indices_o[1]<width_o-1+0.5)
        condition_down_nn  = (indices_o[0]>height_o-1) & (indices_o[0]<height_o-1+0.5) & (indices_o[1]>-0.5) & (indices_o[1]<width_o-1+0.5)

        # concatenate the single conditions using or-statements
        finalcondition = np.where((condition_up_nn | condition_left_nn | condition_right_nn | condition_down_nn))
        indices_o_nn = indices_o.T[finalcondition].T
        indices_n_nn = indices_n.T[finalcondition].T

        # perform nearest neighbour interpolation
        img_t[indices_n_nn[0],indices_n_nn[1]]=img[roundd(indices_o_nn[0]),roundd(indices_o_nn[1])]

    return (img_t, (-1)*np.array([[displacement[0]],[displacement[1]]]))
