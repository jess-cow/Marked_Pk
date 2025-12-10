import jax.numpy as jnp
from jax import grad, jit, vmap
from jax import random
import jax
import nbodykit
import numpy as np
from nbodykit.lab import *
import matplotlib.pyplot as plt
from matplotlib import colors, cm
import matplotlib
import os
import time

def Fourier(field, Nmesh = 256, lbox=700, inverse=False):
    '''Return FFT of field

    
    Parameters:
     field - the 3D field array 
     lbox - Length of simulation in Mpc/h
     Nmesh - size of mesh
     kmin - min wave number to calculate Pk at
     nbins - number of k bins to use
     
    Returns:
     k-centers - grid of k values
     k-field - FFT of real field  
      '''

    # cell width
    d = lbox / Nmesh
    # Take the Fourier Transform of the 3D box
    if inverse:
        complex_field = jnp.fft.irfftn(field, )
    else:
        complex_field = jnp.fft.rfftn(field, )

     # natural wavemodes 
    kx = ky = jnp.fft.fftfreq(Nmesh, d=d)* 2. * np.pi # h/Mpc
    kz = jnp.fft.rfftfreq(Nmesh, d=d)* 2. * np.pi # h/Mpc

    nx, ny, nz = complex_field.shape #shape of ft is symmetric 
    
#     print('fourier',nx,ny,nz)
#     print('kx ky kz', len(kx), len(ky), len(kz))

     # Compute the total wave number
    ktot = jnp.sqrt(kx[:,None, None]**2 + ky[None, :, None]**2+kz[None,None,:]**2)[:nx, :ny, :nz]
#     print(np.shape(ktot), 'ktot')
    if np.isnan(complex_field).any():
        print('fourier transform is nan!')
        quit()
    return ktot, complex_field



def mark_R(p, b, delta_R, R=10):
    '''calculates the White Mark function with smoothing scale of R MPc
    Parameters:
    p
    b
    delta_R : smoothed field
    R: smoothing scale, default 10Mpc
    '''
    m = (1 + (delta_R-1)/(1+b))**(-p)
    return(m)

           
def get_Pk(fourier_data, ktot, second=None,lbox = 700, Nmesh = 256, nbins=144, kmin=0.01, kmax=None):
    """
    Get 1D power spectrum from an already painted field. <- need to do this without NBodykit?
    
    
    Parameters:
     field - the 3D field array 
     lbox - Length of simulation in Mpc/h
     Nmesh - size of mesh
     kmin - min wave number to calculate Pk at
     nbins - number of k bins to use
     
    Returns:
     k-centers - 1D array of central k-bin values
     Pk - 1D array of k at each k 
     n_cells- Number of k-modes in each bin
      
    """
    

    if second is None:
        second = fourier_data
    # else:
        # print('calculating cross Pk')
   
    # Compute the squared magnitude of the complex numbers in the Fourier space, will give 3D Pk array
    power_spectrum =  (fourier_data)*jnp.conjugate(second) #abs changes value here!
#     print('power spec shape', np.shape(power_spectrum))
#     print('ktot', np.shape(ktot))
    if np.isnan(power_spectrum).any():
        print('ERROR: power spectrum is nan')
        print('power spectrum is nan', power_spectrum)
        print('fourier data', fourier_data)
        print('SECOND ', second)
        print('conjugate', np.conjugate(second))
        quit()
    

    # cell width
    d = lbox / Nmesh

    # nyquist frequency k=pi*Nmesh/Lbox, will be the max k in k binning 
    if kmax==None:
        kN = jnp.pi / d
    else:
        kN = kmax
    # print('Kn', kN)
    

    #bin the k to find the total at each |k|
    n_cells, k_bins= jnp.histogram(ktot, bins=Nmesh//2,
                          range=[0, kN], )
    
#     #power spectrum is average so weight by each 3D Pk value and sum,
#     sum_pk, k_bins= jnp.histogram(ktot, bins=nbins,range=[kmin, kN], weights=power_spectrum)  #range=[0.01, 10]
        
    # P(k) estimator david
    sum_pk, k_bins= np.histogram(ktot, bins=Nmesh//2,
                          range=[0, kN],
                          weights=np.real(power_spectrum))
    
    pk = jnp.real(sum_pk/n_cells) #then divide by number averaged over
    
    #find center of k bins
    k_center =  (k_bins[1:] + k_bins[:-1])*.5
    # print('k_centre',k_bins[2]-k_bins[1], k_bins[9]-k_bins[8])
    # print(len(k_center))
    #     vol =(lbox/(2*jnp.pi))**3 
    vol = Nmesh**6/lbox**3
    # /(2*jnp.pi)#in (Mpc/h)^3 #not sure why need 2pi but we do 
    # if np.isnan(pk/vol).any():
    #     print(f"Pk /vol is nan! sum_pk = {sum_pk}, n_cells ={n_cells}, vol is {vol}")

#     from jax.lib import xla_bridge
#     print(xla_bridge.get_backend().platform)  
    return k_center, n_cells, pk/vol

def smooth_field(k, field, R):
    '''
    Smooth a field by a given radius R(Mpc).
    Field supplied should be in k space
    
    '''
    W =  jnp.exp(-0.5*(k*R)**2)
    smoothed_field = (W* field)
    return smoothed_field
