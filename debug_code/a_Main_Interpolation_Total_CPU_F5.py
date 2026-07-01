# -*- coding: utf-8 -*-
"""
Created on Wed Jul 12 11:14:09 2023

@author: itot-lab50 (Dr. Anuj Gupta, CSIR-CSIR, Chandigarh, India)
"""

# 2022, Optics and Laser in Engineering  --  "Fully analytic shading model with specular reﬂections for polygon-based hologram"

import time
t = time.localtime()
current_time = time.strftime("%H:%M:%S", t)
print("Current time is:", current_time)

num_for_ave = 5;                # number of times you want to run program to take average time
time_exe = [];        time_ini = []

      
for _ in range(num_for_ave): 
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
    
    # %% 
    """ Variable declaration """
    lam = 532.e-6;                  lam_by2 = lam/2               
    z = 30.		                    # diffraction distance         
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
    phi_x = 70;                     phi_y = 85  					# incident with x- and y-axis
    
    # Ambient, Diffuse and Specular Reflection Factors
    Ka = 0.2;                       Kd = 0.8;                       Ks = 0.8                          
    Ns = 25                         # Specular (must be odd number) - shininess constant
    
    
    # %% incident light
    lx = np.cos(phi_x*np.pi/180)        
    ly = np.cos(phi_y*np.pi/180)       
    lz = np.sqrt(1-lx**2-ly**2)         
    
    L = np.matrix([lx, ly, lz]).getH()
    
    # %% reflection light
    view = np.matrix([0, 0, 1]).getH()
    view = view/np.linalg.norm(view);    
    H = view+L;                         H = H/np.linalg.norm(H)
    
    #%% Define a certain method :  method = "_proposed" 
    FH = np.zeros((Ny,Nx), dtype = complex)      
    
    #%% 
    """ Defining all the functions here """
    
    #%%
    # Function to read an object   - read .obj data to 'vertex' 'index' and so on
    # This function doesn't read the data with texture information (UV coordinates)
    
    def readObj(file_path, keys):
        data = {}  # Dictionary to store the column values for each unique text in the 1st column
        data[keys[0]] = [];    data[keys[1]] = [];     data[keys[2]] = []
        data['f1'] = [];       data['f2'] = []
        with open(file_path, "r") as file:
            for line in file:
                columns = re.split(r'\s+|//', line.strip())
                #columns = line.strip().slit()  # Split the line by whitespace (vertex only) and // (in faces)
    
                if len(columns) > 0 and columns[0] == keys[0]:
                    data[keys[0]].append([float(val) for val in columns[1:]])
                    #print([float(val) for val in columns[1:]])
                elif len(columns) > 0 and columns[0] == keys[1]:
                    data[keys[1]].append([float(val) for val in columns[1:]])
                elif len(columns) > 0 and columns[0] == keys[2]:
                    data['f1'].append([float(columns[i]) for i in [1,3,5]])
                    data['f2'].append([float(columns[i]) for i in [2,4,6]])
                    #print([float(columns[i]) for i in [1,3,5]])
        return (data)
    
    
    #%%
    # Function to rotate an object
    
    def rotateObj(vertex, vertex_nor,rot_angle):
        eex=rot_angle[0];           eey=rot_angle[1];      eez=rot_angle[2]
        
        #%%
        T1=np.array([[1, 0, 0],[0, np.cos(eex*np.pi/180), -np.sin(eex*np.pi/180)],[0, np.sin(eex*np.pi/180), np.cos(eex*np.pi/180)]])
        T2=np.array([[np.cos(eey*np.pi/180), 0, np.sin(eey*np.pi/180)],[0, 1, 0],[-np.sin(eey*np.pi/180), 0, np.cos(eey*np.pi/180)]])
        T3=np.array([[np.cos(eez*np.pi/180), -np.sin(eez*np.pi/180), 0],[np.sin(eez*np.pi/180), np.cos(eez*np.pi/180), 0],[0, 0, 1]])
        
        
        vertex = np.dot(vertex, np.matrix(np.linalg.multi_dot([T1,T2,T3])).getH())              
        vertex_nor = np.dot(vertex_nor, np.matrix(np.linalg.multi_dot([T1,T2,T3])).getH())
        
        return (vertex, vertex_nor) 
    
    
    #%%
    # Function to resize an object
    
    def resizeObj(vertex,vertex_nor,Lx0,Ly0):
        # scaling object size to match the hologram size
        max1 = np.max(vertex,axis=0).T;      min1 = np.min(vertex,axis=0).T;       # maximum and minimum values along each dimension of the vertices
        xmax = max1[0]-min1[0];              ymax = max1[1]-min1[1]                # calculates the original size of the object in the x and y dimensions
        
        #zmax = max1[2]-min1[2]
        
        scal_x = (0.9*Lx0)/xmax;              scal_y = (0.9*Ly0)/ymax            
        # proportion of scaling - computed to fit the object within 90% of the hologram size in each dimension
        
        scal = np.min([scal_x,scal_y])*0.9
        # minimum is taken and further reduced to 90% to ensure that the object fits within the specified hologram dimensions
        
        if vertex_nor.size > 0:
            vertex_nor = vertex_nor * 1 / scal   
    
        vertex[:,0]=vertex[:,0]*scal;                                              # Different scaling can be done to different axes 
        vertex[:,1]=vertex[:,1]*scal;   
        vertex[:,2]=vertex[:,2]*scal
        
        vertex[:,0]=vertex[:,0]-(np.max(vertex[:,0])+np.min(vertex[:,0]))/2        
        vertex[:,1]=vertex[:,1]-(np.max(vertex[:,1])+np.min(vertex[:,1]))/2     
        # The vertices are translated to the center of the hologram by subtracting the mean of the maximum and minimum values along each dimension.       
        vertex[:,2]=vertex[:,2]-np.min(vertex[:,2])                            # z axis is taken to minimum 
        
        return (vertex, vertex_nor) 
    
    
    #%% backface culling    -  It determines whether a polygon of a graphical object is drawn
    # It is a technique used in computer graphics to improve rendering performance by discarding ("culling") faces that are not visible from the viewpoint.
    
    def backfaceCulling(vertex,index_face,index_nor):
        new_column = np.zeros((index_face.shape[0], 1))
        index_face = np.concatenate((index_face, new_column), axis=1)   
        # Add additional column to store information about marking of the faces to be culled. 
        
        for i in range (index_face.shape[0]):                   
            # shape[0] gives the number of rows of the array - loop iterates over each face in index_face.
            
            V = np.matrix(vertex[index_face[i,:3].astype(int)-1,:]).getH()    
            # The vertex coordinates of each face (i) are extracted from the vertex array based on the face indices.
    
            A = V[:,0];  B = V[:,1]; C = V[:,2];        # vertices coordinates: A, B, C 
    
            AB = np.matrix(B-A).getH();                 AC = np.matrix(C-A).getH()      #  vectors AB and AC
            n = np.cross(AB,AC);                        n = n/np.linalg.norm(n);
            # The face normal n is computed as the cross product of AB and AC and normalized.
            
            if n[:,2]<np.cos(88.9*np.pi/180):       
                # the triangles whose normal direction almost perpendicular to z-axis are considered a back face (angle>88.9).           
                index_face[i,3] = -1      # the corresponding entry in the additional column of index_face is set to -1, marking the back face for culling.
                continue
        
        # Delete the rows from the matrix which satisfy the condition of -1 i.e. corresponds to back face
        index_nor = np.delete(index_nor, index_face[:,3] == -1, axis = 0)          
        index_face = np.delete(index_face, index_face[:,3] == -1, axis = 0)
    
        index_face = np.delete(index_face, 3, axis = 1)     # Delete 4th column from the matrix
        return(index_face, index_nor)
    
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
    
    #%% reading the data from given .obj file
    result_data = readObj(filename, keys)
    
    vertex_ori =  np.array(result_data[keys[0]])           # Vertex normals are vectors perpendicular to the surfaces at each vertex and 
    vertex_nor_ori = np.array(result_data[keys[1]])        # are often used in computer graphics for shading calculations.
    
    index_face_ori = np.array(result_data['f1'])
    index_nor_ori = np.array(result_data['f2'])
    
    
    # %% rotating the 3D object around original point
    vertex_ori_rot, vertex_nor_ori_rot  =  rotateObj(vertex_ori, vertex_nor_ori,rot_angle)
    
    
    #%% scaling size - resize a 3D object represented by its vertices (vertex) and vertex normals (vertex_nor). 
    # The resizing is performed to match the object's size to a specified hologram size. 
    vertex, vertex_nor  =  resizeObj(vertex_ori_rot,vertex_nor_ori_rot, Lx0,Ly0)
    vertex[:,2] = vertex[:,2] + z                           # Diffraction distance is added
    
    # %% back face culling
    index_face, index_nor = backfaceCulling(vertex,index_face_ori,index_nor_ori) 
    index_face = index_face.astype(int) - 1
    index_nor = index_nor.astype(int) - 1
    
    
    #%% Main program starts here
    tf_ini = time.time() - start_time1
    time_ini.append(tf_ini) 

    print("initialization time is: ", tf_ini)
    
    start_time2 = time.time()
    
    for i in range(index_face.shape[0]):
        # %% normal to the faces are calculated here
        nA = vertex_nor[index_nor[i, 0]]            
        nB = vertex_nor[index_nor[i, 1]]
        nC = vertex_nor[index_nor[i, 2]]
        # norms = np.array([norm(nA), norm(nB), norm(nC)])
        nV = np.vstack((nA, nB, nC)).T / np.array([norm(nA), norm(nB), norm(nC)])
        
        # %% Mother triangle properties
        V = np.matrix(vertex[index_face[i,0:3], :]).getH()            # Vertex coordinates
        A = V[:,0];     B = V[:,1];         C = V[:,2];               # vertices coordinates: A, B, C 
    
        AB = np.matrix(B-A).getH();         AC = np.matrix(C-A).getH()      #  vectors AB and AC
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
        
    
        # %% using RectBivariateSpline function in which kx and ky represents degree like 1 for linear, 3 for cubic interpolation
        interp_real = RectBivariateSpline(fy_pb[:, 0][::-1], fx_pb[0, :], F_pb.real[::-1, :], kx=3, ky=3)
        interp_imag = RectBivariateSpline(fy_pb[:, 0][::-1], fx_pb[0, :], F_pb.imag[::-1, :], kx=3, ky=3)
        F_p_real = interp_real(v,u, grid=False)
        F_p_imag = interp_imag(v,u, grid=False)
        F_p = F_p_real + 1j * F_p_imag
    
        # %% Further calculation after interpolated matrices
        Jr = R[0, 0] * R[1, 1] - R[0, 1] * R[1, 0]       
        E1 = np.exp(-1j * twoPi * (np.array(cent[0]) * fx0 + np.array(cent[1]) * fy0 - np.array(cent[2]) * lam_by2 * pre))
    
        FH += Jr*E1*F_p    
    
    # NUFFT: inverse transform: band-limited frequency
    H0 = fftshift(ifft2(fftshift(FH)));       
    H0_crop = H0[int(Nx0/2):int(Nx0*3/2),int(Ny0/2):int(Ny0*3/2)]              # complex hologram  

    tf_exe = time.time() - start_time2
    time_exe.append(tf_exe) 
    
    print('Execution Time is', tf_exe,"s\n")
    
    
    # %% Reconstruction using ASM
    iH = np.pad(H0_crop, ((int(Ny0/2), int(Ny0/2)), (int(Nx0/2), int(Nx0/2))), mode='constant');        
    di = np.max(vertex[:,2])
    
    iF1 = fftshift(fft2(iH))*np.exp(1j*k*di*(1 - lam**2/2 * (fx0**2+fy0**2)))   # transfer function
    
    Ii = ifft2(fftshift(iF1));                            
    Ii_crop = np.abs(Ii[int(Ny0/2):int(Ny0*3/2),int(Nx0/2):int(Nx0*3/2)])
    Ii_norm = (Ii_crop - np.min(Ii_crop)) / (np.max(Ii_crop) - np.min(Ii_crop))

print("Initialization time is", np.mean(time_ini), "\t\t and execution time is", np.mean(time_exe))

plt.imsave(f'Py_IMF5_CPU_Fig1_Complex_Hologram_faces{index_face.shape[0]}_IniTime{round(np.mean(time_ini),3)}_ExeTime{round(np.mean(time_exe),3)}.jpg',np.abs(H0_crop), cmap='gray')

plt.imsave(f'Py_IMF5_CPU_Fig2_Reconstructed_Image_faces{index_face.shape[0]}_IniTime{round(np.mean(time_ini),3)}_ExeTime{round(np.mean(time_exe),3)}.jpg',Ii_norm, cmap='gray')
# Plotting the reconstruction image
plt.imshow(Ii_norm, cmap='gray')
plt.title('Reconstructed Image')
plt.show()

cv2.destroyAllWindows()

import winsound             # Alarm to inform you that program is finished
frequency = 2500            # Set Frequency To 2500 Hertz
duration = 1000             # Set Duration To 1000 ms == 1 second
winsound.Beep(frequency, duration)
