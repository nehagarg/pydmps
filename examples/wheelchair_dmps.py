"""
Copyright (C) 2016 Travis DeWolf

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <http://www.gnu.org/licenses/>.
"""

import numpy as np
import matplotlib.pyplot as plt

import pydmps.dmp_discrete
beta = 20.0 / np.pi
gamma = 100
R_halfpi = np.array(
    [
        [np.cos(np.pi / 2.0), -np.sin(np.pi / 2.0)],
        [np.sin(np.pi / 2.0), np.cos(np.pi / 2.0)],
    ]
)
def avoid_obstacles(y, dy, goal,obstacles):
    p = np.zeros(2)

    dphi_tot = 0
    num_obstacles = 0
    for obstacle in obstacles:
        # based on (Hoffmann, 2009)
        # if we're moving
        if np.linalg.norm(dy) > 1e-5:

            # get the angle we're heading in
            phi_dy = -np.arctan2(dy[1], dy[0])
            R_dy = np.array(
                [[np.cos(phi_dy), -np.sin(phi_dy)], [np.sin(phi_dy), np.cos(phi_dy)]]
            )
            # calculate vector to object relative to body
            obj_vec = obstacle - y
            # rotate it by the direction we're going
            obj_vec = np.dot(R_dy, obj_vec)
            # calculate the angle of obj relative to the direction we're going
            phi = np.arctan2(obj_vec[1], obj_vec[0])

            dphi = gamma * phi * np.exp(-beta * abs(phi))
            R = np.dot(R_halfpi, np.outer(obstacle - y, dy))
            pval = -np.nan_to_num(np.dot(R, dy) * dphi)
            #pval = -np.nan_to_num(np.dot(R, dy) )
            # check to see if the distance to the obstacle is further than
            # the distance to the target, if it is, ignore the obstacle
            #if np.linalg.norm(obj_vec) > np.linalg.norm(goal - y):
            #    pval = 0
            #    dphi = 0
            #else:
            num_obstacles += 1
            #print(dphi)
            #print(R)
            #print(pval)
            p += pval
            dphi_tot += dphi
    print(dphi_tot)
    if num_obstacles > 0:
        print(num_obstacles)
        p /= num_obstacles*1.0
        dphi_tot /= num_obstacles*1.0
    print(p)
    return p,dphi_tot
def read_trajectories(file_name):
    
    h = open(file_name, 'r')

    # Reading from the file
    content = h.readlines()
    content_counter = 0
    num_goals = int(content[content_counter])
    content_counter = content_counter + 1
    goal_positions = []
    all_intermediate_goals = []
    
    #print(num_goals)
    for i in range(num_goals):
        goal_pos_x, goal_pos_y = [float(x) for x in content[content_counter].split()]
        content_counter = content_counter + 1
        goal_positions.append([goal_pos_x, goal_pos_y])

        num_intermediate_goals = int(content[content_counter])
        content_counter = content_counter + 1
        intermediate_goals = [[0.0,0.0]]
        for j in range(num_intermediate_goals):
            goal_pos_x, goal_pos_y, goal_pos_yaw = [float(x) for x in content[content_counter].split()]
            content_counter = content_counter + 1
            intermediate_goals.append([goal_pos_x, goal_pos_y])
        all_intermediate_goals.append(np.array(intermediate_goals))
        
        #plt.plot(all_intermediate_goals[i][:, 0], all_intermediate_goals[i][:, 1], color=plt.cm.RdYlBu(i), lw=2, alpha=0.5)
        plt.plot(all_intermediate_goals[i][:, 0], all_intermediate_goals[i][:, 1], "b--", lw=2, alpha=0.5)
    #print(goal_positions)
    return all_intermediate_goals

def read_lidar(file_name):
    h = open(file_name, 'r')

    # Reading from the file
    content = h.readlines()
    content_counter = 0
    num_lidar_points = int(content[content_counter])
    content_counter = content_counter + 1
    lidar_points = []
    for i in range(num_lidar_points):
        lidar_x,lidar_y = [float(x) for x in content[content_counter].split()]
        content_counter = content_counter + 1
        lidar_points.append([lidar_x, lidar_y])
    lidar_points_np = np.array(lidar_points)
    if num_lidar_points > 0:
        plt.scatter(lidar_points_np[:, 0], lidar_points_np[:, 1], c="r")
    return lidar_points_np    

def trim_lidar_points(trajectory, lidar_points_np):
    lidar_points_trimmed = []
    for lidar_point in lidar_points_np :
        if np.linalg.norm(lidar_point - trajectory[0]) < 3:
            lidar_points_trimmed.append([lidar_point[0],lidar_point[1]])
    return np.array(lidar_points_trimmed)

def imitate_trajectory(y_des, trajectory_id, lidar_points_np):
    #y_des = np.load("2.npz")["arr_0"].T
    #y_des -= y_des[:, 0][:, None]

    # test normal run
    dmp = pydmps.dmp_discrete.DMPs_discrete(n_dmps=2, n_bfs=500, ay=np.ones(2) * 10.0)
    print(y_des)
    dmp.imitate_path(y_des=y_des)
    print(dmp.w[0])
    print(dmp.w[1])
    plt.figure(2+trajectory_id, figsize=(6, 6))

    y_track, dy_track, ddy_track = dmp.rollout()
    plt.plot(y_track[:, 0], y_track[:, 1], "b--", lw=2, alpha=0.5)

    # run while moving the target up and to the right
    y_track = []
    dmp.reset_state()
    dmp.y += np.array([0.5, 0.5])
    for t in range(dmp.timesteps):
        #print(dmp.goal)
        #p,dphi_tot = avoid_obstacles(dmp.y, dmp.dy, dmp.goal, lidar_points_np)
        #y, _, _ = dmp.step(external_force = p)
        y, _, _ = dmp.step()
        
        print(y)
        y_track.append(np.copy(y))
        # move the target slightly every time step
        #dmp.goal += np.array([1e-2, 1e-2])
    y_track = np.array(y_track)

    plt.plot(y_track[:, 0], y_track[:, 1], "b", lw=2)
    if len(lidar_points_np) > 0:
        plt.scatter(lidar_points_np[:, 0], lidar_points_np[:, 1], c="r")
    #plt.plot(lidar_points_np[:, 0], lidar_points_np[:, 1], "r--", lw=2, alpha=0.5)
    plt.title("DMP system - draw number 2")

    plt.axis("equal")
    #plt.xlim([-2, 2])
    #plt.ylim([-2, 2])
    plt.legend(["original path", "moving target"])
    plt.show()




def get_dmp(dt_ = 0.01):
    # test normal run
    dmp = pydmps.dmp_discrete.DMPs_discrete(n_dmps=2, n_bfs=500, dt = dt_, ay=np.ones(2) * 10.0)
    print("Inside Python")
    print(dmp.c)
    print(dmp)
    #dmp.imitate_path(y_des=y_des)
    return dmp
def dmp_print(dmp):
    print("Inside Print DMP")
    print(dmp)
    return dmp
def imitate_path(dmp, y_des_pylist):
    y_des_np = (np.array(y_des_pylist)).T
    print(y_des_np)
    print(dmp)
    dmp.imitate_path(y_des=y_des_np)
    print("Imitated path")
    print(dmp.c.shape)
    print(dmp.h.shape)
    print(dmp.w.shape)
    #print(dmp.w[0])
    #print(dmp.w[1])
    return dmp
def dmp_rollout(dmp):
    print("Inside python dmp_rollout function")
    plt.figure(1, figsize=(6, 6))

    y_track, dy_track, ddy_track = dmp.rollout()
    plt.plot(y_track[:, 0], y_track[:, 1], "b--", lw=2, alpha=0.5)
    print(y_track)
    # run while moving the target up and to the right
    y_track = []
    dmp.reset_state()
    dmp.y += np.array([0.5, 0.5])
    for t in range(dmp.timesteps):
        #print(dmp.goal)
        
        y, _, _ = dmp.step()
        y_track.append(np.copy(y))
        # move the target slightly every time step
        #dmp.goal += np.array([1e-2, 1e-2])
    y_track = np.array(y_track)

    plt.plot(y_track[:, 0], y_track[:, 1], "b", lw=2)
    plt.title("DMP system - from python")

    plt.axis("equal")
    plt.xlim([-2, 2])
    plt.ylim([-2, 2])
    plt.legend(["original path", "moving target"])
    plt.show()


def dmp_plot(y_des_,y_track_):
    print("Inside python dmp_plot function")
    y_des = np.array(y_des_)
    y_track = np.array(y_track_)
    print(y_des)
    print(y_track)
    plt.figure(1, figsize=(6, 6))
    
    #y_track, dy_track, ddy_track = dmp.rollout()
    plt.plot(y_des[:, 0], y_des[:, 1], "b--", lw=2, alpha=0.5)

    plt.plot(y_track[:, 0], y_track[:, 1], "b", lw=2)
    plt.title("DMP system - from C++")

    plt.axis("equal")
    plt.xlim([-2, 2])
    plt.ylim([-2, 2])
    plt.legend(["original path", "moving target"])
    plt.show()


def display_stored_trajectory_dmps(u_range_index = 759):
    for i in range(35,u_range_index):
        print(i)
        print(i)
        plt.figure(1, figsize=(6, 6))
        data_number = str(i)
        all_intermediate_goals = read_trajectories("/home/neha/WORK_FOLDER/code/ntu-rris/catkin_ws_shared_control_pomdp/data/Voronoi_paths_step_" + data_number  + ".txt")
        lidar_points_np = read_lidar("/home/neha/WORK_FOLDER/code/ntu-rris/catkin_ws_shared_control_pomdp/data/Lidar_" + data_number + ".txt")
        lidar_points_trimmed = trim_lidar_points(all_intermediate_goals[0], lidar_points_np)
        if len(lidar_points_trimmed) > 0:
            plt.show()
            #for j in range(len(all_intermediate_goals)):
            for j in range(1):
                imitate_trajectory(all_intermediate_goals[j].T,j,lidar_points_trimmed)
        else:
            print("not going in if")
    
if __name__ == "__main__":
    display_stored_trajectory_dmps(36)