import numpy as np
import matplotlib.pyplot as plt
import math as mth
import h5py
from plotter_funcs import *
from scipy import ndimage

sim_sz=60           #Size of simulation in physical units Mpc/h cubed
grid_nodes=128      #Density Field grid resolution
smooth_scl=2.0      #Smoothing scale in physical units Mpc/h
tot_mass_bins=4     #Number of Halo mass bins
particles_filt=300  #Halos to filter out based on number of particles, ONLY for Dot Product Spin-LSS(SECTION 5.)
Mass_res=1.35*10**8 #Bolchoi particle mass as per, https://arxiv.org/pdf/1002.3660.pdf

#Load Bolchoi Simulation Catalogue, ONLY filtered for X,Y,Z=<sim_sz
f=h5py.File("bolchoi_DTFE_rockstar_box_%scubed_xyz_vxyz_jxyz_m_r.h5"%sim_sz, 'r')
data=f['/halo'][:]#data array: (Pos)XYZ(Mpc/h), (Vel)VxVyVz(km/s), (Ang. Mom)JxJyJz((Msun/h)*(Mpc/h)*km/s), (Vir. Mass)Mvir(Msun/h) & (Vir. Rad)Rvir(kpc/h) 
f.close()

Xc=data[:,0]#halo X coordinates
Yc=data[:,1]#halo Y coordinates
Zc=data[:,2]#halo Z coordinates
h_mass=data[:,9]#halo Virial Mass
halos=np.column_stack((Xc,Yc,Zc))

# SECTION 1. Density field creation ------------------------------------------------
#Manual technique to bin halos within a 3D Matrix
Xc_min=np.min(Xc)
Xc_max=np.max(Xc)
Yc_min=np.min(Yc)
Yc_max=np.max(Yc)
Zc_min=np.min(Zc)
Zc_max=np.max(Zc)

Xc_mult=grid_nodes/(Xc_max-Xc_min)
Yc_mult=grid_nodes/(Yc_max-Yc_min)
Zc_mult=grid_nodes/(Zc_max-Zc_min)

Xc_minus=Xc_min*grid_nodes/(Xc_max-Xc_min)+0.0000001
Yc_minus=Yc_min*grid_nodes/(Yc_max-Yc_min)+0.0000001
Zc_minus=Zc_min*grid_nodes/(Zc_max-Zc_min)+0.0000001

image_orgnl=np.zeros((grid_nodes,grid_nodes,grid_nodes))
for i in range(len(Xc)):
   #Create index related to the eigenvector bins
    grid_index_x=mth.trunc(halos[i,0]*Xc_mult-Xc_minus)      
    grid_index_y=mth.trunc(halos[i,1]*Yc_mult-Yc_minus) 
    grid_index_z=mth.trunc(halos[i,2]*Zc_mult-Zc_minus)   
    image_orgnl[grid_index_x,grid_index_y,grid_index_z]+=h_mass[i]#Add halo mass to coinciding pixel 
    
#END SECTION-------------------------------------------------------------------------
    
s=1.0*smooth_scl/sim_sz*grid_nodes# s- standard deviation of Kernel, converted from Mpc/h into number of pixels
#smooth via ndimage function
img = ndimage.gaussian_filter(image_orgnl,s,order=0,mode='wrap',truncate=20)#smoothing function

#-----------------------
# Gaussian smooth PENG METHOD
#-----------------------
image=np.fft.fftn(image_orgnl)
rc=1.*sim_sz/(grid_nodes)
### creat k-space grid
kx = range(grid_nodes)/np.float64(grid_nodes)

for i in range(grid_nodes/2+1, grid_nodes):
    kx[i] = -np.float64(grid_nodes-i)/np.float64(grid_nodes)

kx = kx*2*np.pi/rc
ky = kx
kz = kx

kx2 = kx**2
ky2 = ky**2
kz2 = kz**2

# smooth_scl**2
Rs2 = np.float64(smooth_scl**2)/2.

print 'do smoothing in k-space will be much easier ....'
for i in range(grid_nodes):
    for j in range(grid_nodes):
        for k in range(grid_nodes):
            index = kx2[i] + ky2[j] + kz2[k]
            # smoothing in k-space
            image[i,j,k] = np.exp(-index*Rs2)*image[i,j,k]#convolving 1 pixel at a time

# transform to real space and save it
smoothed_image = np.fft.ifftn(image).real
#---------------------------------

'''
hf = h5py.File('CIC_density_smoothed.h5', 'r')
smoothed_image_peng=hf['CIC_density_smoothed'][:]
hf.close()
smoothed_image_peng=np.reshape(smoothed_image_peng,(grid_nodes,grid_nodes,grid_nodes))
'''
#plotting
slc=30
plt.figure(figsize=(15,17))
col=2
ro=2
ax1=plt.subplot2grid((col,ro), (0,1)) 
plt.title('ndimage')
cmapp = plt.get_cmap('jet')
scl_plt=5#reduce scale of density fields and eigenvalue subplots by increasing number
dn_fl_plt=ax1.imshow(np.power(img[slc,:,:],1.0/scl_plt),cmap=cmapp,extent=[0,grid_nodes,0,grid_nodes])#The colorbar will adapt to data
plt.colorbar(dn_fl_plt,cmap=cmapp)
ax2=plt.subplot2grid((col,ro), (1,0)) 
plt.title('Peng')
scl_plt=5#reduce scale of density fields and eigenvalue subplots by increasing number
dn_fl_plt=ax2.imshow(np.power(smoothed_image[slc,:,:],1.0/scl_plt),cmap=cmapp,extent=[0,grid_nodes,0,grid_nodes])#The colorbar will adapt to data
plt.colorbar(dn_fl_plt,cmap=cmapp)

ax3=plt.subplot2grid((col,ro), (0,0)) 
plt.title('density field')
scl_plt=5#reduce scale of density fields and eigenvalue subplots by increasing number
dn_fl_plt=ax3.imshow(np.power(image_orgnl[slc,:,:],1.0/scl_plt),cmap=cmapp,extent=[0,grid_nodes,0,grid_nodes])#The colorbar will adapt to data
plt.colorbar(dn_fl_plt,cmap=cmapp)





