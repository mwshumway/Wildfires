

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
    a = np.pi/8 #angle, may be affected by variance of wind
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
    sigma2_0 (ndarray): Initial variance
    k (float): Growth rate in direction of wind
    c (float): Growth rate in direction perpendicular to wind

    Returns:
    float: Fire intensity at time t
    """
    mu_x = x0 + w1 * t
    mu_y = y0 + w2 * t
    sigma_parallel = sigma2_0[0] + k * t
    sigma_perp = sigma2_0[1] + c * t

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
    sigma_parallel = sigma2_0[0] + k * t
    sigma_perp = sigma2_0[1] + c * t

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





def simulate(wind,fires,survivors, wind_var,tp):
    loc = np.zeros(2)
    s = len(survivors)
    sig0 = np.ones(2)*.1
    s0 = np.array([0, 0, 0, 0])

    solutions = []
    for step in range(s+1):
        if step == s:
            target = np.zeros(2)
        else:
            path = survivors - loc
            dist = np.linalg.norm(path, ord = 2, axis = 1)
    
            #check for danger
            cones = get_cones(fires, wind)
            mask = in_cones(cones, survivors, fires)
            incentive = 5
            dist[mask] -= incentive
            
            target_ind = np.argmin(dist)
            target = survivors[target_ind]


        # Initial mesh and guess
        t_steps = 500
        t = np.linspace(0, 1, t_steps)
        z0 = np.zeros((8, t_steps))
        z0[0, :] = np.linspace(s0[0], target[0], t_steps)
        z0[1, :] = np.linspace(s0[1], target[1], t_steps)
        p0 = np.array([1.0])
        v_guess = (target - s0[:2]) / p0[0]
        time_penalty = tp
        z0[2, :] = v_guess[0]
        z0[3, :] = v_guess[1]

        w1, w2 = wind
        k,c = wind_var
        u_norm = np.sqrt(w1**2 + w2**2)
        u_x = w1 / u_norm
        u_y = w2 / u_norm


        def fire_ode(t, z, p):
            x, y, vx, vy, p1, p2, p3, p4 = z
            tf = np.clip(p[0], 0, None)
            f1_x, f1_y = grad_fire(tf * t, x, y, fires[0][0], fires[0][1],w1,w2,u_x,u_y, sig0, k, c)
            f2_x, f2_y = grad_fire(tf * t, x, y, fires[1][0], fires[1][1],w1,w2,u_x,u_y, sig0, k, c)
            return tf * np.array([ vx,
                                    vy,
                                    0.5 * p3 / control_penalty,
                                    0.5 * p4 / control_penalty,
                                    f1_x + f2_x,
                                    f1_y + f2_y,
                                    -p1,
                                    -p2])


        def bc(ya, yb, p):
            x, y, vx, vy, p1, p2, p3, p4 = yb
            u1, u2 = 0.5 * p3 / control_penalty, 0.5 * p4 / control_penalty
            tf = np.clip(p[0], 0, None)
            fire1 = fire(tf, x, y, fires[0][0],fires[0][1], w1,w2,u_x,u_y,sig0, k, c)
            fire2 = fire(tf, x, y, fires[1][0],fires[1][1], w1,w2,u_x,u_y, sig0, k, c)
            H = p1*vx + p2*vy + p3*u1 + p4*u2 - (time_penalty + fire1 + fire2 + control_penalty * (u1**2 + u2**2))
            return np.array([ya[0] - s0[0],
                            ya[1] - s0[1],
                            ya[2] - s0[2],
                            ya[3] - s0[3],
                            yb[0] - target[0],
                            yb[1] - target[1],
                            yb[2],
                            yb[3],
                            H   ]) 

        # Solve BVP
        sol = solve_bvp(fire_ode, bc, t, z0, p0, tol=1e-2, max_nodes=1_000_00)

        if not sol.success:
            print("BVP solver failed:", sol.message)
            return None

        solutions.append(sol)

        loc = target
        s0 = sol.y[:4,-1]
        tf = abs(sol.p[0])
        fires = fires + wind*tf
        sig0 = sig0 + wind_var*tf
        if len(survivors)>0:
            survivors = np.delete(survivors, target_ind,axis =0)
        print(target)

    return solutions
        


def animate(sol,wind, wind_var, survivors,fires):

    tf = sum([abs(sol[i].p[0]) for i in range(5)])
    print(tf)
    x = np.concatenate([sol[i].y[:2] for i in range(5)], axis = 1)
    x_traj, y_traj = x[0], x[1]
    t_eval = np.linspace(0, tf, len(x_traj))  # Full resolution time vector

    # Create a lower-resolution time vector just for the animation
    n_frames = 200  # Adjust as needed
    t_anim_vals = np.linspace(0, tf, n_frames)

    # Setup plot
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.set_xlim(-3, 12)
    ax.set_ylim(-1, 12)
    ax.set_xlabel('X-axis')
    ax.set_ylabel('Y-axis')
    ax.set_title('2 Wildfire Avoidance, Final Time: {:.2f}'.format(tf))

    # Plot target
    # ax.scatter(survivors[:,0], survivors[:,1], marker='x', s = 100, label = 'Target')
    # # ax.scatter(target[0], target[1], marker='x', color='red', label='Target', s=100)


    dude_img = mpimg.imread("images/lil_guy.png")
    dude_size = 1.2
    for x, y in survivors:
        ax.imshow(
            dude_img,
            extent=[x - dude_size/2, x + dude_size/2, y - dude_size/4, y + 3*dude_size/4],
            zorder=3,
            aspect='auto',
            label = 'Target'
        )

    traj_line, = ax.plot([], [], 'b--', lw=2, label='Agent Trajectory')
    current_pos, = ax.plot([], [], 'bo', markersize=8)
    ax.legend()

    drone_img = mpimg.imread("drone2.webp")
    drone_size = .65
    drone = ax.imshow(drone_img, extent=[-100, -99, -100, -99], zorder=3, aspect='auto', label = 'Agent')

    def update(t_anim):
        global contour
        if contour is not None:
            for coll in contour.collections:
                coll.remove()


        w1,w2 = wind
        k,c = wind_var
        u_norm = np.sqrt(w1**2 + w2**2)
        u_x = w1 / u_norm
        u_y = w2 / u_norm
        sigma_0 = 1

        ax.set_title(f'Two Wildfires, Four Targets\ntime: {round(t_anim,3)}')

        # Fire center positions at this frame
        mean1 = [fires[0][0] + w1 * t_anim, fires[0][1] + w2 * t_anim]
        mean2 = [fires[1][0] + w1 * t_anim, fires[1][1] + w2 * t_anim]
        sigma_parallel = sigma_0 + k * t_anim
        sigma_perp = sigma_0 + c * t_anim

        Sigma = np.array([
            [u_x**2 * sigma_parallel + u_y**2 * sigma_perp, 
             u_x * u_y * (sigma_parallel - sigma_perp)],
            [u_x * u_y * (sigma_parallel - sigma_perp), 
             u_y**2 * sigma_parallel + u_x**2 * sigma_perp]
        ])

        # lilx1 = np.linspace(mean1[0]- 3*Sigma[0,0], mean1[0] + 3*Sigma[0,0]) 
        # lily1 = np.linspace(mean1[1]- 3*Sigma[1,1], mean1[1] + 3*Sigma[1,1])
        # lilx2 = np.linspace(mean2[0]- 3*Sigma[0,0], mean2[0] + 3*Sigma[0,0])
        # lily2 = np.linspace(mean2[1]- 3*Sigma[1,1], mean2[1] + 3*Sigma[1,1])

        # X,Y = np.meshgrid(np.concatenate((lilx1, lilx2)), np.concatenate((lily1, lily2)))
        X,Y = np.meshgrid(np.linspace(-3,12),np.linspace(-1,12))
        # Compute the fire intensity heatmap
        Z1 = multivariate_normal.pdf(np.dstack((X, Y)), mean=mean1, cov=Sigma)
        Z2 = multivariate_normal.pdf(np.dstack((X, Y)), mean=mean2, cov=Sigma)
        Z = Z1 + Z2

        contour = ax.contourf(X, Y, Z, levels=20, cmap='hot', alpha=0.65)

        # Update trajectory up to current time
        mask = t_eval <= t_anim
        traj_line.set_data(x_traj[mask], y_traj[mask])
        if mask.any():
            x_curr, y_curr = x_traj[mask][-1], y_traj[mask][-1]
            current_pos.set_data([x_traj[mask][-1], y_traj[mask][-1]])
            drone_extent = [
            x_curr - drone_size, x_curr + drone_size,
            y_curr - drone_size, y_curr + drone_size
        ]

        drone.set_extent(drone_extent)

        return []

    # Animate using downsampled frames
    ani = animation.FuncAnimation(
        fig, update, frames=t_anim_vals, blit=False, interval=50, repeat=True
    )

    # Save animation
    ani.save('animations/so_many_people_to_save.mp4', writer='ffmpeg', fps=25)

        

if __name__ == "__main__":
    #inintial conditions and constants
    wind = np.array([.2,.15])
    survivors = np.array([[10,4],
                          [3,9],
                          [9,4],
                          [8,10]])

    
    fires = np.array([[1,4],
                      [6,0]])

    wind_var = np.array([.3,.1])

    control_penalty = 5
    fire_cost = 52_500
    contour = None
    tp = 75

    sol = simulate(wind, fires, survivors, wind_var,tp)

    if sol:
        animate(sol,wind,wind_var,survivors,fires)


    # cones = get_cones(fires,wind)

    # plt.scatter(fires[:,0], fires[:,1], marker = 'o', s = 100)
    # plt.scatter(survivors[:,0], survivors[:,1],marker='x')
    # plt.plot([fires[0,0],cones[0][0,0]], [fires[0,1], cones[0][1,0]])
    # plt.plot([fires[0,0],cones[0][0,1]], [fires[0,1], cones[0][1,1]])

    # plt.plot([fires[1,0],cones[1][0,0]], [fires[1,1], cones[1][1,0]])
    # plt.plot([fires[1,0],cones[1][0,1]], [fires[1,1], cones[1][1,1]])

    # x = np.concatenate([sol[i].y[:2] for i in range(3)], axis = 1)

    # plt.plot(x[0], x[1])
    # # plt.plot(sol[1].y[0], sol[1].y[1])
    # # plt.plot(sol[2].y[0], sol[2].y[1])
    
    # plt.ylim(-1,12)
    # plt.xlim(-1,12)
    # plt.show()