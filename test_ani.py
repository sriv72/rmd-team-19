#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr  2 10:09:28 2024

@author: jakemcgrath
"""

import numpy as np
from scipy.optimize import fsolve
import matplotlib.pyplot as plt
import imageio

def get_points(t1,t2,t3,t4,t5):
    [Ax,Ay] = [0, 0]
    [Bx,By] = [l6, 0]
    [Cx,Cy] = [Bx - l23*np.cos(t1), -l23*np.sin(t1)]
    [Dx,Dy] = [Cx - l12*np.cos(t1), Cy - l12*np.sin(t1)]
    [Ex,Ey] = [Dx + l2*np.cos(t2), Dy + l2*np.sin(t2)]
    [Fx,Fy] = [Cx + l4*np.cos(t4), Cy + l4*np.sin(t4)]
    [Gx,Gy] = [Dx - l23*np.cos(t1) , Dy - l23*np.sin(t1)]
    return([Ax,Ay,Bx,By,Cx,Cy,Dx,Dy,Ex,Ey,Fx,Fy,Gx,Gy])

# Generate frames (plots)
l12 = 1;
l23 = 2.93;
l2 = 2.71;
l3 = 1.643;
l4 = 2.07;
l5 = 1.789;
l6 = 5.445;
num_frames = 250
theta_values = np.concatenate((np.linspace(0.25, 0.94, num_frames), np.linspace(0.94, 0.25, num_frames)), axis=0)

Cx_history = []
Cy_history = []
Dx_history = []
Dy_history = []
Ex_history = []
Ey_history = []
Fx_history = []
Fy_history = []
Gx_history = []
Gy_history = []

# Function to generate plot for a given frame
def generate_plot(theta):
    t1 = theta
    def equations(x):
        t2, t3, t4, t5 = x
        e1x = l6 - l23 * np.cos(t1) + l4 * np.cos(t4) - l3 * np.cos(t3) + l5 * np.cos(t5)
        e1y = -l23 * np.sin(t1) + l4 * np.sin(t4) - l3 * np.sin(t3) + l5 * np.sin(t5)
        e2x = l3 * np.cos(t3) - l4 * np.cos(t4) - l12 * np.cos(t1) + l2 * np.cos(t2)
        e2y = l3 * np.sin(t3) - l4 * np.sin(t4) - l12 * np.sin(t1) + l2 * np.sin(t2)
        return [e1x, e1y, e2x, e2y]
    
    t2_guess = (3.352 + 2.360) / 2
    t3_guess = (0.267 + 0.446) / 2
    t4_guess = (3.320 + 2.086) / 2
    t5_guess = (0.961 + 2.346) / 2
    initial_guess = [t2_guess, t3_guess, t4_guess, t5_guess]
    solution = fsolve(equations, initial_guess)
    t2 = (solution[0]%(2 * np.pi))
    t3 = (solution[1]%(2 * np.pi))
    t4 = (solution[2]%(2 * np.pi))
    t5 = (solution[3]%(2 * np.pi))
    [Ax,Ay,Bx,By,Cx,Cy,Dx,Dy,Ex,Ey,Fx,Fy,Gx,Gy] = get_points(t1,t2,t3,t4,t5)
    
    Cx_history.append(Cx)
    Cy_history.append(Cy)
    Dx_history.append(Dx)
    Dy_history.append(Dy)
    Ex_history.append(Ex)
    Ey_history.append(Ey)
    Fx_history.append(Fx)
    Fy_history.append(Fy)
    Gx_history.append(Gx)
    Gy_history.append(Gy)
    
    # Create plot
    plt.figure()
    
    plt.scatter(Ax,Ay,s=50,c='k',label='A',zorder=10)
    plt.scatter(Bx,By,s=50,c='k',label='B',zorder=10)
    plt.scatter(Cx,Cy,s=50,c='k',label='C',zorder=10)
    plt.scatter(Dx,Dy,s=50,c='k',label='D',zorder=10)
    plt.scatter(Ex,Ey,s=50,c='k',label='E',zorder=10)
    plt.scatter(Fx,Fy,s=50,c='k',label='F',zorder=10)
    plt.scatter(Gx,Gy,s=50,c='k',label='G',zorder=10)
    
    plt.plot([Ax,Bx],[Ay,By],linewidth=4,color='black')
    plt.plot([Bx,Cx],[By,Cy],linewidth=4,color='red')
    plt.plot([Cx,Dx],[Cy,Dy],linewidth=4,color='red')
    plt.plot([Dx,Ex],[Dy,Ey],linewidth=4,color='blue')
    plt.plot([Ex,Fx],[Ey,Fy],linewidth=4,color='green')
    plt.plot([Fx,Cx],[Fy,Cy],linewidth=4,color='orange')
    plt.plot([Ex,Ax],[Ey,Ay],linewidth=4,color='purple')
    plt.plot([Gx,Dx],[Gy,Dy],linewidth=4,color='red')
    
    plt.plot(Cx_history, Cy_history, color='gray', alpha=0.5)
    plt.plot(Dx_history, Dy_history, color='gray', alpha=0.5)
    plt.plot(Ex_history, Ey_history, color='gray', alpha=0.5)
    plt.plot(Fx_history, Fy_history, color='gray', alpha=0.5)
    plt.plot(Gx_history, Gy_history, color='gray', alpha=0.5)
    
    plt.xlim([-2,6])
    plt.ylim([-6,1])
    
    plt.gca().set_aspect('equal', adjustable='box')
    t_temp = np.rad2deg(theta)
    plt.title(f'θ1: {t_temp:.2f}')
    plt.xlabel('X')
    plt.ylabel('Y')
    
    # Save plot as image
    plt.savefig(f'frame_{theta:.2f}.png')
    plt.close()

# Generate plots for each frame
for theta in theta_values:
    generate_plot(theta)

# Read saved images and create GIF
images = []
for theta in theta_values:
    images.append(imageio.imread(f'frame_{theta:.2f}.png'))

# Save images as GIF
imageio.mimsave('animation_trail.gif', images, fps=50)