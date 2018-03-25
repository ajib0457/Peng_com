import numpy as np
import matplotlib.pyplot as plt
import math as mth
import h5py
from plotter_funcs import *
from scipy import ndimage

sim_sz=60           #Size of simulation in physical units Mpc/h cubed
grid_nodes=128      #Density Field grid resolution
smooth_scl=2.0      #Smoothing scale in physical units Mpc/h

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

#3d smoothing via convolution method
in_val,fnl_val=-grid_nodes/2.0,grid_nodes/2.0#kernel values
X,Y,Z=np.meshgrid(np.linspace(in_val,fnl_val,grid_nodes),np.linspace(in_val,fnl_val,grid_nodes),np.linspace(in_val,fnl_val,grid_nodes))
h=(1/np.sqrt(1.0*2*np.pi*s*s))**(3)*np.exp(-1/(1.0*2*s*s)*((Y-0.5)**2+(X-0.5)**2+(Z-0.5)**2))
#h=(1/sqrt(1.0*2*pi*s*s))**(3)*exp(-1/(1.0*2*s*s)*((Y-0.5)**2+(X-0.5)**2+(Z-0.5)**2))
area=np.sum(h)
h=h/area
h=np.roll(h,int(grid_nodes/2),axis=0)
h=np.roll(h,int(grid_nodes/2),axis=1)
h=np.roll(h,int(grid_nodes/2),axis=2)
fft_dxx=np.fft.fftn(h)
fft_db=np.fft.fftn(image_orgnl)
ifft_dxx=np.fft.ifftn(np.multiply(fft_dxx,fft_db)).real

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

#calculate difference between smoothed fields
my_mns_peng=np.sum(abs(ifft_dxx.flatten()-smoothed_image.flatten()))
my_mns_ndimag=np.sum(abs(ifft_dxx.flatten()-img.flatten()))

ndimag_mns_peng=np.sum(abs(img.flatten()-smoothed_image.flatten()))
ndimag_mns_my=np.sum(abs(img.flatten()-ifft_dxx.flatten()))

peng_mns_my=np.sum(abs(smoothed_image.flatten()-ifft_dxx.flatten()))
peng_mns_ndimag=np.sum(abs(smoothed_image.flatten()-img.flatten()))

#plotting
plt.figure(figsize=(15,17))
#Plotting global features
slc=15

columns=['Density Values']
subtitl_offset=1.15
ttl_fnt=12
plt.subplots_adjust(hspace=0.3,wspace=0.12,top=0.85)
plt.suptitle("Density fld smoothing comparison.\nSmoothing scale: %s pxls \nSlice: %s/%s"%(round(s,4),slc,grid_nodes),y=0.95,fontsize=20)
col=2
ro=2

ax1=plt.subplot2grid((col,ro), (0,1)) 
plt.title('ndimage func',y=subtitl_offset,fontsize=ttl_fnt)
cmapp = plt.get_cmap('jet')
scl_plt=5#reduce scale of density fields and eigenvalue subplots by increasing number
dn_fl_plt=ax1.imshow(np.power(img[slc,:,:],1.0/scl_plt),cmap=cmapp,extent=[0,grid_nodes,0,grid_nodes])#The colorbar will adapt to data
plt.colorbar(dn_fl_plt,cmap=cmapp)
#generate sub-table
data_fnc=np.round(np.array([[ndimag_mns_peng],[ndimag_mns_my]]),8)
rows=['ndimag - peng','ndimag - my']
tbl_diff=plt.table(cellText=data_fnc,loc='top',rowLabels=rows,colLabels=columns,colWidths=[0.5 for x in columns],cellLoc='center')
tbl_diff.set_fontsize(12)
tbl_diff.scale(1.2, 1.2)

ax2=plt.subplot2grid((col,ro), (1,0)) 
plt.title('Peng method',y=subtitl_offset,fontsize=ttl_fnt)
scl_plt=5#reduce scale of density fields and eigenvalue subplots by increasing number
dn_fl_plt=ax2.imshow(np.power(smoothed_image[slc,:,:],1.0/scl_plt),cmap=cmapp,extent=[0,grid_nodes,0,grid_nodes])#The colorbar will adapt to data
plt.colorbar(dn_fl_plt,cmap=cmapp)
#generate sub-table
data_fnc=np.round(np.array([[peng_mns_my],[peng_mns_ndimag]]),8)
rows=['peng - my','peng - ndimag']
tbl_diff=plt.table(cellText=data_fnc,loc='top',rowLabels=rows,colLabels=columns,colWidths=[0.5 for x in columns],cellLoc='center')
tbl_diff.set_fontsize(12)
tbl_diff.scale(1.2, 1.2)

ax3=plt.subplot2grid((col,ro), (0,0)) 
plt.title('density field')
scl_plt=5#reduce scale of density fields and eigenvalue subplots by increasing number
dn_fl_plt=ax3.imshow(np.power(image_orgnl[slc,:,:],1.0/scl_plt),cmap=cmapp,extent=[0,grid_nodes,0,grid_nodes])#The colorbar will adapt to data
plt.colorbar(dn_fl_plt,cmap=cmapp)

ax3=plt.subplot2grid((col,ro), (1,1)) 
plt.title('convolution method',y=subtitl_offset,fontsize=ttl_fnt)
scl_plt=5#reduce scale of density fields and eigenvalue subplots by increasing number
dn_fl_plt=ax3.imshow(np.power(ifft_dxx[slc,:,:],1.0/scl_plt),cmap=cmapp,extent=[0,grid_nodes,0,grid_nodes])#The colorbar will adapt to data
plt.colorbar(dn_fl_plt,cmap=cmapp)
#generate sub-table
data_fnc=np.round(np.array([[my_mns_peng],[my_mns_ndimag]]),8)
rows=['my - peng','my - ndimag']
tbl_diff=plt.table(cellText=data_fnc,loc='top',rowLabels=rows,colLabels=columns,colWidths=[0.5 for x in columns],cellLoc='center')
tbl_diff.set_fontsize(12)
tbl_diff.scale(1.2, 1.2)


plt.savefig('/import/oth3/ajib0457/wang_peng_code/Peng_run_bolchoi_smpl_python/smoothed_comp_plt/bolchoi_smpl%sMpch_smth%sMpch_grid%s.png'%(sim_sz,smooth_scl,grid_nodes))

