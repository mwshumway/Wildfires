

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from scipy.stats import multivariate_normal
from scipy.integrate import solve_bvp
import matplotlib.image as mpimg
from scipy import ndimage
from PIL import Image



def get_cones(fires, wind):
    #predict fire travel
    t = [0,10]
    a = np.pi/16 #angle, may be affected by variance of wind
    initial_plane = np.array([[np.cos(-a), np.cos(a)],
                              [np.sin(-a), np.sin(a)]])
    theta = -np.arctan2(wind[1], wind[0])
    rotate = np.array([[np.cos(theta), np.sin(theta)],
                        [-np.sin(theta), np.cos(theta)]])
    
    plane = np.array( [ti* rotate@initial_plane for ti in t])
    
    cones = []
    for f in fires:
        cones.append(np.concatenate((f.reshape(-1,1),f.reshape(-1,1)), axis = 1) + plane[-1] - plane[0])
        
    return cones

def in_cones(cones, survivors,fires):
    danger = []
    for s in survivors:
        boolean = False
        for i,c in enumerate(cones):
            if np.cross(c[:,0]-fires[i], s-fires[i]) >= 0 and np.cross(s-fires[i], c[:,1]-fires[i]) >= 0:
                boolean = True
        danger.append(boolean)

    return np.array(danger)




fire_cost = 50_000 # Cost of fire

def fire(t, x, y, x0, y0, w1, w2, u_x, u_y, sigma2_0, k, c):
    """
    Fire dynamics model.
    
    Parameters:
    t (float): Time
    x (float): X-coordinate
    y (float): Y-coordinate
    x0 (float): Initial x-coordinate of fire
    y0 (float): Initial y-coordinate of fire
    w1 (float): Wind speed in x-direction
    w2 (float): Wind speed in y-direction
    u_x (float): Wind direction in x (unit vector)
    u_y (float): Wind direction in y (unit vector)
    sigma2_0 (float): Initial variance
    k (float): Growth rate in direction of wind
    c (float): Growth rate in direction perpendicular to wind

    Returns:
    float: Fire intensity at time t
    """
    mu_x = x0 + w1 * t
    mu_y = y0 + w2 * t
    sigma_parallel = sigma2_0 + k * t
    sigma_perp = sigma2_0 + c * t

    # Covariance matrix elements for each t
    cov_11 = u_x**2 * sigma_parallel + u_y**2 * sigma_perp
    cov_12 = u_x * u_y * (sigma_parallel - sigma_perp)
    cov_22 = u_y**2 * sigma_parallel + u_x**2 * sigma_perp 

    # Compute determinant and inverse elements
    det = cov_11 * cov_22 - cov_12**2
    if np.any(det <= 0):
        raise ValueError("Determinant of covariance matrix is non-positive.")

    inv_11 = cov_22 / det
    inv_12 = -cov_12 / det
    inv_22 = cov_11 / det

    dx = x - mu_x
    dy = y - mu_y

    # Compute exponent
    exponent = -0.5 * (inv_11 * dx**2 + 2 * inv_12 * dx * dy + inv_22 * dy**2)
    normalization = 1.0 / (2 * np.pi * np.sqrt(det))
    return fire_cost * normalization * np.exp(exponent)

def grad_fire(t, x, y, x0, y0, w1, w2, u_x, u_y, sigma2_0, k, c):
    """
    Gradient of fire intensity.
    
    Parameters:
    t (float): Time
    x (float): X-coordinate
    y (float): Y-coordinate
    x0 (float): Initial x-coordinate of fire
    y0 (float): Initial y-coordinate of fire
    w1 (float): Wind speed in x-direction
    w2 (float): Wind speed in y-direction
    u_x (float): Wind direction in x (unit vector)
    u_y (float): Wind direction in y (unit vector)
    sigma2_0 (float): Initial variance
    k (float): Growth rate in direction of wind
    c (float): Growth rate in direction perpendicular to wind

    Returns:
    np.ndarray: Gradient of fire intensity
    """
    f = fire(t, x, y, x0, y0, w1, w2, u_x, u_y, sigma2_0, k, c)
    
    mu_x = x0 + w1 * t
    mu_y = y0 + w2 * t
    sigma_parallel = sigma2_0 + k * t
    sigma_perp = sigma2_0 + c * t

    # Covariance matrix elements
    cov_11 = u_x**2 * sigma_parallel + u_y**2 * sigma_perp
    cov_12 = u_x * u_y * (sigma_parallel - sigma_perp)
    cov_22 = u_y**2 * sigma_parallel + u_x**2 * sigma_perp 

    # Compute inverse elements
    det = cov_11 * cov_22 - cov_12**2
    if np.any(det <= 0):
        raise ValueError("Determinant of covariance matrix is non-positive.")
    
    inv_11 = cov_22 / det
    inv_12 = -cov_12 / det
    inv_22 = cov_11 / det

    dx = x - mu_x
    dy = y - mu_y

    grad_x = -f * (inv_11 * dx + inv_12 * dy)
    grad_y = -f * (inv_12 * dx + inv_22 * dy)
    
    return grad_x, grad_y




def simulate(wind,fires,survivors):
    loc = np.zeros(2)
    s = len(survivors)
    for _ in range(s):
        path = survivors - loc
        dist = np.linalg.norm(path, ord = 2, axis = 1)
 
        #check for danger
        cones = get_cones(fires, wind)
        mask = in_cones(cones, survivors, fires)
        incentive = 5
        dist[mask] -= incentive
        
        target = np.argmin(dist)
        print(target)

        
        

if __name__ == "__main__":
    wind = np.array([2,1])
    survivors = np.array([[10,4],
                          [3,9],
                          [10,5]])
    
    fires = np.array([[2,4],
                      [6,1]])
    
    simulate(wind, fires, survivors)
    cones = get_cones(fires,wind)

    plt.scatter(fires[:,0], fires[:,1], marker = 'o', s = 100)
    plt.scatter(survivors[:,0], survivors[:,1],marker='x')
    plt.plot([fires[0,0],cones[0][0,0]], [fires[0,1], cones[0][1,0]])
    plt.plot([fires[0,0],cones[0][0,1]], [fires[0,1], cones[0][1,1]])

    plt.plot([fires[1,0],cones[1][0,0]], [fires[1,1], cones[1][1,0]])
    plt.plot([fires[1,0],cones[1][0,1]], [fires[1,1], cones[1][1,1]])
    plt.ylim(-1,12)
    plt.xlim(-1,12)
    plt.show()