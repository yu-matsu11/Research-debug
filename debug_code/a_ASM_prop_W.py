# -*- coding: utf-8 -*-
"""
Created on Mon Dec 16 15:15:17 2024

@author: wangfan
"""

import numpy as np
from numpy.fft import fft2, ifft2, fftshift, ifftshift
import cv2
import matplotlib.pyplot as plt


# iH =  ASM_prop(H0, A[2,0], dp, lam, radius=0.5)

        
def ASM_prop(input, di, pp, wlen, radius=None):
    if radius is not None:
        FH = fftshift(fft2(input))

        filter_mask = circleW(FH.shape, radius)


        FH = FH * filter_mask
        input = ifft2(ifftshift(FH))
        
    Ny0 = len(input);
    Nx0 = len(input[0]);
    input_pad = np.pad(input, ((int(Ny0/2), int(Ny0/2)), (int(Nx0/2), int(Nx0/2))), mode='constant'); # pading two times
    
    Ny = len(input_pad);
    # print("Ny=", Ny)
    Nx = len(input_pad[0]);    
    # print("Nx=",Nx)
    k = 2*np.pi/wlen;
    Lx = Nx * pp;
    Ly = Ny * pp;
    fx = np.linspace(-1/pp/2,(1/pp/2)-(1/Lx),Nx,endpoint=True); 
    fy = np.linspace(-1/pp/2,(1/pp/2)-(1/Ly),Ny,endpoint=True);
    fx,fy = np.meshgrid(fx,fy);
    fx_BL = Nx/2 * pp/wlen/ np.abs(di);  # band-limited angular spectrum propagation
    fy_BL = Ny/2 * pp/wlen/ np.abs(di);
    fx[np.abs(fx)>fx_BL] = 0;
    fy[np.abs(fy)>fy_BL] = 0;
    
    di = np.ceil(di/wlen)*wlen;
    transfer = np.exp(1j * k * di * np.sqrt(1 - (wlen * fx)**2 - (wlen*fy)**2));
    # print(len(transfer))
    # print(len(transfer[0]))
    fH = fftshift(fft2(fftshift(input_pad))) * transfer;
    
    H = fftshift(ifft2(fftshift(fH)));
    # print(Ny0)
    # print(Ny)
    # print(Nx0)
    # print(Nx)
    H = H[int(Ny/2)-int(Ny0/2)+1:int(Ny/2)+int(Ny0/2),int(Nx/2)-int(Nx0/2)+1:int(Nx/2)+int(Nx0/2)];
    return H


def circleW(shape, radi):
    """
    Create a circular mask.
    radi: relative radius (0~1, relative to half width)
    """
    r, c = shape
    y, x = np.ogrid[:r, :c]
    cx, cy = c/2, r/2
    radius = (c * radi) / 2
    mask = (x - cx)**2 + (y - cy)**2 <= radius**2
    return mask.astype(float)
    

    
## perform propagation for an image    
# img_dir_path = r"image_path.png" # the address of input image
# img = cv2.imread(img_dir_path)
# img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY);
# # plt.imshow(img, cmap='gray')
# # plt.show()

# H = ASM_prop(img,15,8.e-3,530.e-6);  ## mm as unit
# dphase = dphase.dphase(H);
# # phase  = phase/phase.max()*255.0;

# plt.imsave(img_dir_path+"_double phase hologram2.png" ,dphase, cmap='gray')
