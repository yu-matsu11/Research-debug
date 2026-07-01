# -*- coding: utf-8 -*-
"""
Created on Wed Jul 12 11:14:09 2023
@author: itot-lab50 (Dr. Anuj Gupta, CSIR-CSIR, Chandigarh, India)

updated on 2025-11-11
@author: Fan Wang

"""

# 2022, Optics and Laser in Engineering  --  "Fully analytic shading model with specular reﬂections for polygon-based hologram"

import time
t = time.localtime()
current_time = time.strftime("%H:%M:%S", t)
print("Current time is:", current_time)

time_exe = [];        time_ini = []

      
# %%
start_time1 = time.time()
""" importing required libraries """
import numpy as np
from numpy.fft import fft2, ifft2, fftshift
from numpy.linalg import norm, det
from scipy.interpolate import RectBivariateSpline
import matplotlib.pyplot as plt
import re
import cv2
from a_ASM_prop_W import ASM_prop
from a_interp2_W import interp2

# %% 
""" Variable declaration """
lam = 532.e-6;                  lam_by2 = lam/2               
z = 50.		                    # diffraction distance         
twoPi = 2*np.pi;                k = twoPi/lam;

Nx0 = 1024;                     Ny0 = 1024            # number of sampling 
Nx = 2*Nx0;                     Ny = 2*Ny0            # double zero-pading or not

dp = 3.74e-3;                   dp2 = 2*dp            # pixel pitch in SLM plane
Lx0 = Nx0*dp;                   Ly0 = Ny0*dp          # The dimensions of the hologram along x and y-axes

# load the cooridnates data of 3D object 
filename = "teapot1860.obj";    keys = ["v", "vn", "f"];        rot_angle = [10., -50., -30.]       


# %% Frequency coordinates defining 
fx0d = np.linspace(-1/(dp2),(1/(dp2))-(1/(2*Lx0)),Nx,endpoint=True)          
fy0d = np.linspace(1/(dp2),(-1/(dp2))+(1/(2*Ly0)),Ny,endpoint=True)            # linearly spaced vector
fx0,fy0 = np.meshgrid(fx0d,fy0d)         
             
pre = (fx0**2+fy0**2)
       
# %% illumination parameters
phi_x = 70;                    
phi_y = 85  					# incident with x- and y-axis

# Ambient, Diffuse and Specular Reflection Factors
Ka = 0.2;                      
Kd = 0.8;                       
Ks = 0.8                          
Ns = 25                         # Specular (must be odd number) - shininess constant


# %% incident light
lx = np.cos(phi_x*np.pi/180)        
ly = np.cos(phi_y*np.pi/180)       
lz = np.sqrt(1-lx**2-ly**2)         

L = np.matrix([lx, ly, lz]).T

# %% reflection light
view = np.matrix([0, 0, 1]).T
view = view/np.linalg.norm(view);    
H = view+L;                         H = H/np.linalg.norm(H)

#%% Define a certain method :  method = "_proposed" 
FH = np.zeros((Ny,Nx), dtype = complex)      

#%% 
""" Defining all the functions here """

#%%
# Function to read an object   - read .obj data to 'vertex' 'index' and so on
# This function doesn't read the data with texture information (UV coordinates)

#%% Rotation Matrix in Interpolation - rotates any vector around the normal vector 'n' given as the argument
def RotMatrix_W(n):
    ez = np.array([0, 0, 1])
    cos_theta = np.inner(ez, n);          
    sin_theta = np.linalg.norm(np.cross(ez, n))

    nn = np.array([n[0], n[1], 0])     # Only x and y components of n vector are taken like the projection on xy-plane
    nn = nn / np.linalg.norm(nn)
    ex = np.array([1, 0, 0])    
    cos_phi = np.inner(ex, nn);           
    sin_phi = np.linalg.norm(np.cross(ex, nn))

    if n[1] < 0:
        sin_phi = -sin_phi
    # The following R is obtained by multiplying 2 rotation matrices corresponding to theta and phi angles
    R = np.array([[cos_phi*cos_theta, cos_theta*sin_phi, -sin_theta],
                  [-sin_phi, cos_phi, 0],
                  [cos_phi*sin_theta, sin_phi*sin_theta, cos_theta]])

    if abs(n[2]-1) < 1e-6:
        R = np.eye(3)
    return (R)

def plot_tri_new(Vr, dp, nV):
    Lx_p = np.max(Vr[0, :]) - np.min(Vr[0, :])
    Ly_p = np.max(Vr[1, :]) - np.min(Vr[1, :])

    sft = np.zeros(2)
    sft[0] = Lx_p / 2 - np.max(Vr[0, :])
    sft[1] = Ly_p / 2 - np.max(Vr[1, :])
    Vr = Vr + np.array([[sft[0]], [sft[1]], [0]])

    Nx_p = np.ceil(Lx_p / dp);      Ny_p = np.ceil(Ly_p / dp)
    Nx_p += Nx_p % 2;               Ny_p += Ny_p % 2
    Lx_p = Nx_p * dp;               Ly_p = Ny_p * dp

    x = np.linspace(-Lx_p / 2, Lx_p / 2 - dp, int(Nx_p))
    y = np.linspace(Ly_p / 2, -Ly_p / 2 + dp, int(Ny_p))
    x, y = np.meshgrid(x, y)
    grid = np.zeros((int(Ny_p), int(Nx_p), 2))

    grid[:, :, 0] = x;              grid[:, :, 1] = y

    Ap = Vr[:, 0];                  Bp = Vr[:, 1];         Cp = Vr[:, 2]

    OAOB = np.array((Ap[0] - x)) * np.array((Bp[1] - y)) - np.array((Bp[0] - x)) * np.array((Ap[1] - y))                
    # This np.array is added to change data type 
    OBOC = np.array((Bp[0] - x)) * np.array((Cp[1] - y)) - np.array((Cp[0] - x)) * np.array((Bp[1] - y))
    OCOA = np.array((Cp[0] - x)) * np.array((Ap[1] - y)) - np.array((Ap[0] - x)) * np.array((Cp[1] - y))
    
    lgc_AB = (OAOB >= 0).astype(int);          lgc_BC = (OBOC >= 0).astype(int)
    lgc_CA = (OCOA >= 0).astype(int);          lgc = lgc_AB + lgc_BC + lgc_CA
    triangle = (lgc == 0) + (lgc == 3)
    
    L_p = np.array([Lx_p, Ly_p]);               N_p = np.array([Nx_p, Ny_p])

    # %% solve the pixel normal by barycentric coordinates
    mat = np.array([[1, 1, 1], [Vr[0, 0], Vr[0, 1], Vr[0, 2]], [Vr[1, 0], Vr[1, 1], Vr[1, 2]]])
    mat = np.array(mat, dtype=float)                # This line of changing data type to float is added
    triarea = abs(1/2 * det(mat))                   # area of the triangle with the determinant of the matrix

    E = lambda M, N, Px, Py: np.array(Px - M[0]) * np.array(N[1] - M[1]) - np.array(Py - M[1]) * np.array(N[0] - M[0])
    lambda_A = E(Bp, Cp, x, y) / triarea / 2;           lambda_B = E(Cp, Ap, x, y) / triarea / 2
    lambda_C = E(Ap, Bp, x, y) / triarea / 2
    idx_neg = (lambda_A + lambda_B + lambda_C < 0)              

    lambda_A[idx_neg] = -lambda_A[idx_neg];             lambda_B[idx_neg] = -lambda_B[idx_neg]
    lambda_C[idx_neg] = -lambda_C[idx_neg]

    # %% interpolation of vertex normals to get the pixel normal
    nA = nV[:, 0];           nB = nV[:, 1];           nC = nV[:, 2];
    norVec = lambda i: nA[i] + lambda_B * np.array(nB[i] - nA[i]) + lambda_C * np.array(nC[i] - nA[i])
    
    nP = np.zeros((int(Ny_p), int(Nx_p), 3))                    
    for i1 in range(3):
        nP[:, :, i1] = norVec(i1)

    lenVec = np.sqrt(nP[:, :, 0]**2 + nP[:, :, 1]**2 + nP[:, :, 2]**2)
    nP[:, :, 0] /= lenVec          # Normalized
    nP[:, :, 1] /= lenVec;         nP[:, :, 2] /= lenVec

    return (triangle, L_p, N_p, sft, grid, nP)

# %% 
""" Function declaration done  -  Now call the functions and execute the task """

#%% Main program starts here
tf_ini = time.time() - start_time1
time_ini.append(tf_ini) 

print("initialization time is: ", tf_ini)

start_time2 = time.time()

# %% normal to the faces are calculated here
nA = np.matrix([0.2871647 ,  0.42727871,  1.31792246])  #vertex_nor[index_nor[i, 0]]            
nB = np.matrix([-0.73134422,  0.39691713,  1.14435131]) 
nC = np.matrix([-0.47076329,  0.17157627,  1.32321243]) 
# norms = np.array([norm(nA), norm(nB), norm(nC)])
nV = np.vstack((nA, nB, nC)).T / np.array([norm(nA), norm(nB), norm(nC)])

# %% Mother triangle properties
V = np.matrix([[-5.7762e-04, -8.0161e-04, -6.2709e-05],
    [ -2.9079e-04,  3.7071e-04,  8.5896e-04],
    [ 1.3139e-01,  1.5139e-01,  1.2143e-01]])*10**3 # [x,y,z].T

# V = np.matrix([[-5.7762e-04, -8.0161e-04, -6.2709e-05],
#     [ -2.9079e-04,  3.7071e-04,  8.5896e-04],
#     [ 1.3139e-04,  1.5139e-04,  1.2143e-04]])*10**3 # [x,y,z].T


A = V[:,0];     B = V[:,1];         C = V[:,2];               # vertices coordinates: A, B, C 

AB = (B-A).T;         AC = (C-A).T      #  vectors AB and AC
n = np.cross(AB,AC).ravel();        n = n/np.linalg.norm(n)         # Face normal

# %% Interpolation Section starts here
R = RotMatrix_W(n)                              # Global-to-local coordinate transform
cent = np.mean(V, axis=1)
Vr = np.dot(R, V - cent.reshape(-1, 1))

tri, L_p, N_p, sft, grid, nP = plot_tri_new(Vr, dp, nV)

# Calculate shading values
diffuse = np.array(L[0]) * nP[:, :, 0] + np.array(L[1]) * nP[:, :, 1] + np.array(L[2]) * nP[:, :, 2]
specular = np.array(H[0]) * nP[:, :, 0] + np.array(H[1]) * nP[:, :, 1] + np.array(H[2]) * nP[:, :, 2]
tri_shading = (Ka + diffuse * Kd + (specular ** Ns) * Ks) * tri

tri_shading = np.pad(tri_shading, ((int(N_p[1]) // 2, int(N_p[1]) // 2), (int(N_p[0]) // 2, int(N_p[0]) // 2)), 'constant')

# %% Rotated frequency definition
u = R[0, 0] * fx0 + R[0, 1] * fy0 - R[0, 2] * lam_by2 * pre                # rotation frequency cooridinates
v = R[1, 0] * fx0 + R[1, 1] * fy0 - R[1, 2] * lam_by2 * pre

fx_pb = np.linspace(-1/(dp2), 1/(dp2) - 1/(L_p[0]*2), int(N_p[0]*2))     # double sampling coordinates
fy_pb = np.linspace(1/(dp2), -1/(dp2) + 1/(L_p[1]*2), int(N_p[1]*2))
fx_pb, fy_pb = np.meshgrid(fx_pb, fy_pb)

F_pb = fftshift(fft2(fftshift(tri_shading)))
F_pb *= np.exp(1j * twoPi * (sft[0] * fx_pb + sft[1] * fy_pb))

# interpolate
x, y, pts, F_p = interp2(fx_pb, fy_pb, F_pb, u, v, method='linear', fill_value=0)

# %% Further calculation after interpolated matrices
Jr = R[0, 0] * R[1, 1] - R[0, 1] * R[1, 0]       
E1 = np.exp(-1j * twoPi * (np.array(cent[0]) * fx0 + np.array(cent[1]) * fy0 - np.array(cent[2]) * lam_by2 * pre))

FH = Jr*E1*F_p    
# FH = FH[int(Nx0/2):int(Nx0*3/2),int(Ny0/2):int(Ny0*3/2)]  
plt.figure(1)
plt.imshow(abs(FH),'gray')
plt.title("spectral distribution")




# hologram
H0 = fftshift(ifft2(fftshift(FH)));       
H0 = H0[int(Nx0/2):int(Nx0*3/2),int(Ny0/2):int(Ny0*3/2)]              # complex hologram  



tf_exe = time.time() - start_time2
time_exe.append(tf_exe) 

print('Execution Time is', tf_exe,"s\n")

# %% Reconstruction using ASM
iH =  ASM_prop(H0, C[2, 0], dp, lam, radius=0.5)
# print(A[2, 0])


plt.figure(2)
plt.imshow(np.log(abs(iH)),'gray')
plt.title("reconstruction image")
plt.show()