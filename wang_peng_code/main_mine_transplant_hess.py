import numpy as np
import matplotlib.pyplot as plt
from sympy import *
import math as mth
import h5py
import sklearn.preprocessing as skl
from plotter_funcs import * 

sim_sz=250           #Size of simulation in physical units Mpc/h cubed
grid_nodes=200      #Density Field grid resolution
smooth_scl=3.5      #Smoothing scale in physical units Mpc/h
tot_mass_bins=5    #Number of Halo mass bins
particles_filt=300  #Halos to filter out based on number of particles, ONLY for Dot Product Spin-LSS(SECTION 5.)
Mass_res=1.35*10**8 #Bolchoi particle mass as per, https://arxiv.org/pdf/1002.3660.pdf

#Load Bolchoi Simulation Catalogue, ONLY filtered for X,Y,Z=<sim_sz
#f=h5py.File("bolchoi_DTFE_rockstar_box_%scubed_xyz_vxyz_jxyz_m_r.h5"%sim_sz, 'r')
f=h5py.File("bolchoi_DTFE_rockstar_allhalos_xyz_vxyz_jxyz_m_r.h5", 'r')

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

image=np.zeros((grid_nodes,grid_nodes,grid_nodes))
for i in range(len(Xc)):
   #Create index related to the eigenvector bins
    grid_index_x=mth.trunc(halos[i,0]*Xc_mult-Xc_minus)      
    grid_index_y=mth.trunc(halos[i,1]*Yc_mult-Yc_minus) 
    grid_index_z=mth.trunc(halos[i,2]*Zc_mult-Zc_minus)   
    image[grid_index_x,grid_index_y,grid_index_z]+=h_mass[i]#Add halo mass to coinciding pixel 
#END SECTION-------------------------------------------------------------------------
    
# **IGNORE** SECTION 1.1 Plot scatter, velocity & AM vector scatter----------------------------
slc=160             # 0 - grid_nodes  
plane=1             #X-0, Y-1, Z-2   
plane_thickness=5 #Mpc/h
scatter_plot(Xc,Yc,Zc,slc,plane,grid_nodes,plane_thickness,sim_sz)
#-----------------------------------------------------------------------------------
'''
# SECTION 2. Create Hessian ---------------------------------------------------------------------------------------
X,Y,Z,s=symbols('X Y Z s')#sympy feature needed to take derivative of Gaussian 'h'.
h=(1/sqrt(2.0*pi*s*s))**(3)*exp(-1/(2.0*s*s)*((Y-0.5)**2+(X-0.5)**2+(Z-0.5)**2))#Will differentiate this analytically dxx,dxy...

#h=(1/sqrt(1.0*2*pi*s*s))**(3)*exp(-1/(1.0*2*s*s)*((Y-0.5)**2+(X-0.5)**2+(Z-0.5)**2))
#take second partial derivatives as per Hessian 
hprimexx=h.diff(X,X)#dxx
hprimexy=h.diff(X,Y)#dxy
hprimeyy=h.diff(Y,Y)#dyy
hprimezz=h.diff(Z,Z)#dzz
hprimezx=h.diff(Z,X)#dzx
hprimezy=h.diff(Z,Y)#dzy
#Lambdify i.e make them variables once again
fxx=lambdify((X,Y,Z,s),hprimexx,'numpy')
fxy=lambdify((X,Y,Z,s),hprimexy,'numpy')
fyy=lambdify((X,Y,Z,s),hprimeyy,'numpy')
fzz=lambdify((X,Y,Z,s),hprimezz,'numpy')
fzx=lambdify((X,Y,Z,s),hprimezx,'numpy')
fzy=lambdify((X,Y,Z,s),hprimezy,'numpy')
#3d meshgrid for each kernel and evaluate 6 partial derivatives to generate 9 kernels

#Kernal settings
img_x,img_y,img_z=np.shape(image)
s=1.0*smooth_scl/sim_sz*grid_nodes# s- standard deviation of Kernel, converted from Mpc/h into number of pixels
in_val,fnl_val=-grid_nodes/2,grid_nodes/2#Boundary of kernel

#Kernel generator: The above lines from beginning SECTION 2. have analytically taken the second derivative of the 3D gaussian 'h' 
X,Y,Z=np.meshgrid(np.linspace(in_val,fnl_val,img_x),np.linspace(in_val,fnl_val,img_y),np.linspace(in_val,fnl_val,img_z))
dxx=fxx(X,Y,Z,s)#Now generate kernel for each unique hessian term
dxy=fxy(X,Y,Z,s)
dyy=fyy(X,Y,Z,s)
dzz=fzz(X,Y,Z,s)
dzx=fzx(X,Y,Z,s)
dzy=fzy(X,Y,Z,s)
   
#shift kernels to make periodic 
dxx=np.roll(dxx,int(img_x/2),axis=0)
dxx=np.roll(dxx,int(img_y/2),axis=1)
dxx=np.roll(dxx,int(img_z/2),axis=2)
dxy=np.roll(dxy,int(img_x/2),axis=0)
dxy=np.roll(dxy,int(img_y/2),axis=1)
dxy=np.roll(dxy,int(img_z/2),axis=2)
dyy=np.roll(dyy,int(img_x/2),axis=0)
dyy=np.roll(dyy,int(img_y/2),axis=1)
dyy=np.roll(dyy,int(img_z/2),axis=2)
dzz=np.roll(dzz,int(img_x/2),axis=0)
dzz=np.roll(dzz,int(img_y/2),axis=1)
dzz=np.roll(dzz,int(img_z/2),axis=2)
dzx=np.roll(dzx,int(img_x/2),axis=0)
dzx=np.roll(dzx,int(img_y/2),axis=1)
dzx=np.roll(dzx,int(img_z/2),axis=2)
dzy=np.roll(dzy,int(img_x/2),axis=0)
dzy=np.roll(dzy,int(img_y/2),axis=1)
dzy=np.roll(dzy,int(img_z/2),axis=2)
#fft 6 kernels & density field
fft_dxx=np.fft.fftn(dxx)
fft_dxy=np.fft.fftn(dxy)
fft_dyy=np.fft.fftn(dyy)
fft_dzz=np.fft.fftn(dzz)
fft_dzx=np.fft.fftn(dzx)
fft_dzy=np.fft.fftn(dzy)
fft_db=np.fft.fftn(image)
#convolution of kernels with density field & inverse transform
ifft_dxx=np.fft.ifftn(np.multiply(fft_dxx,fft_db)).real
ifft_dxy=np.fft.ifftn(np.multiply(fft_dxy,fft_db)).real
ifft_dyy=np.fft.ifftn(np.multiply(fft_dyy,fft_db)).real
ifft_dzz=np.fft.ifftn(np.multiply(fft_dzz,fft_db)).real
ifft_dzx=np.fft.ifftn(np.multiply(fft_dzx,fft_db)).real
ifft_dzy=np.fft.ifftn(np.multiply(fft_dzy,fft_db)).real
#reshape into column matrices
ifft_dxx=np.reshape(ifft_dxx,(np.size(ifft_dxx),1))      
ifft_dxy=np.reshape(ifft_dxy,(np.size(ifft_dxy),1))
ifft_dyy=np.reshape(ifft_dyy,(np.size(ifft_dyy),1))
ifft_dzz=np.reshape(ifft_dzz,(np.size(ifft_dzz),1))
ifft_dzx=np.reshape(ifft_dzx,(np.size(ifft_dzx),1))
ifft_dzy=np.reshape(ifft_dzy,(np.size(ifft_dzy),1))
#create Hessian
hessian=np.column_stack((ifft_dxx,ifft_dxy,ifft_dzx,ifft_dxy,ifft_dyy,ifft_dzy,ifft_dzx,ifft_dzy,ifft_dzz))  
hessian=np.reshape(hessian,(grid_nodes**3,3,3))
#END SECTION----------------------------------------------------------------------------------------------------
'''
# SECTION 2 PENG TRANSPLANT------------------------------------------------------------------------------
'''
#-----------------------
# FFT the density before do smoothing
#-----------------------
smoothing_cic_density=np.fft.fftn(image)
print len(smoothing_cic_density)
#-----------------------
# Gaussian smooth
#-----------------------
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
            smoothing_cic_density[i,j,k] = np.exp(-index*Rs2)*smoothing_cic_density[i,j,k]


# transfer to real space and save it
smoothed_cic_density = np.fft.ifftn(smoothing_cic_density).real
'''
s=1.0*smooth_scl/sim_sz*grid_nodes# s- standard deviation of Kernel, converted from Mpc/h into number of pixels
img = ndimage.gaussian_filter(image,s,order=0,mode='wrap',truncate=20)#smoothing function
smoothing_cic_density=np.fft.fftn(img)
#-----------------------
# Hessian on each grid
#-----------------------
print 'do Hessian in k-space will be much easier ....'
dens_k = smoothing_cic_density  # the smoothing cic field in k-space

Hessian = np.zeros([grid_nodes,grid_nodes,grid_nodes,6])

array=np.zeros([grid_nodes,grid_nodes,grid_nodes])

for ii in range(6):
    print 'component', ii
    for i in range(grid_nodes):
        for j in range(grid_nodes):
            for k in range(grid_nodes):
                  if ii == 0: a = -kx[i]**2
                  if ii == 1: a = -kx[i]*ky[j]
                  if ii == 2: a = -kx[i]*kz[k]
                  if ii == 3: a = -ky[j]**2
                  if ii == 4: a = -ky[j]*kz[k]
                  if ii == 5: a = -kz[k]**2
                  array[i,j,k] = a

    Hessian[:,:,:,ii] = np.fft.ifftn(dens_k*array).real

#Rearrange and reshape Hessian to suit my format
hessian=np.column_stack((Hessian[:,:,:,0].flatten(),Hessian[:,:,:,1].flatten(),Hessian[:,:,:,2].flatten(),Hessian[:,:,:,0].flatten(),Hessian[:,:,:,3].flatten(),Hessian[:,:,:,4].flatten(),Hessian[:,:,:,2].flatten(),Hessian[:,:,:,4].flatten(),Hessian[:,:,:,5].flatten()))

hessian=np.reshape(hessian,(grid_nodes**3,3,3))
#END SECTION 2-----------------------------------------------------------------------------------------
'''
# **IGNORE** SECTION 2.1 Plot scatter,smoothed, eigvals, classification mask,color scatter, table with halo-classif ratios----------------------------
scl_plt=50 #nth root scaling of density field
smoothed_density_field(grid_nodes,slc,smooth_scl,in_val,fnl_val,s,image,plane,scl_plt)
#-----------------------------------------------------------------------------------
'''
# SECTION 3. Calculate and reorder eigenpairs----------------------------------------------
eig_vals_vecs=np.linalg.eig(hessian)
#Eigenvalues 
eigvals_unsorted=eig_vals_vecs[0]
eigvals=np.sort(eigvals_unsorted)
eig_one=eigvals[:,2]
eig_two=eigvals[:,1]
eig_three=eigvals[:,0]
eigvals_unsorted=eigvals_unsorted.flatten()
#Eigenvectors
eigvecs=eig_vals_vecs[1]
vec_arr_num,vec_row,vec_col=np.shape(eigvecs)
values=np.reshape(eigvecs.transpose(0,2,1),(vec_row*vec_arr_num,vec_col))#orient eigenvectors so that each row is an eigenvector
#END SECTION------------------------------------------------------------------------------
'''
#SECTION 3.1 plot eigenvalues-------------------------------------------
eigenvalue_plts(eig_one,eig_two,eig_three,grid_nodes,sim_sz,smooth_scl)
#------------------------------------------------------------------------
'''

# SECTION 4. Classify Large scale structure--------------------------------------------------------------------
lss=['filament','sheet','void','cluster']#which LSS to output mask & eigenpairs for
def lss_classifier(lss,eigvals_unsorted,values,eig_one,eig_two,eig_three):   
    '''
    This is the classifier function which takes input:
    
    lss: the labels of Large scale structure which will be identified pixel by pixel and also eigenvectors 
    will be retrieved if applicable.
    
    eigvals_unsorted: flattened eigenvalues of entire simulation
    
    values: eigenvectors of entire simulation reshaped into row vectors

    eig_one,two and three: These are the isolated eigenvalues 
    
    This function will output:
    
    eig_fnl: An array containing all of the relevent eigenvectors for each LSS type
    
    mask_fnl: array prescribing 0-void, 1-sheet, 2-filament and 3-cluster
    '''
    eig_fnl=np.zeros((grid_nodes**3,4))#first column will store all relevent eigenvalues and last 3 columns will store eigenvectors
    mask_fnl=np.zeros((grid_nodes**3))
    for i in lss:#Will loop once for each LSS chosen using 'lss' list
        vecsvals=np.column_stack((eigvals_unsorted,values))#Redefine vecsvals for each iteration
        vecsvals=np.reshape(vecsvals,(grid_nodes**3,3,4))#Each pixel is 3x4 where first column is eigenvalues and next 3 col are eigenvectors
        recon_img=np.zeros([grid_nodes**3])#Mask used along with below conditions to mask out pixels 
        #Conditions for each LSS
        if (i=='void'):
            recon_filt_one=np.where(eig_three>0)
            recon_filt_two=np.where(eig_two>0)
            recon_filt_three=np.where(eig_one>0)
        if (i=='sheet'):
            recon_filt_one=np.where(eig_three<0)
            recon_filt_two=np.where(eig_two>=0)
            recon_filt_three=np.where(eig_one>=0)
        if (i=='filament'):
            recon_filt_one=np.where(eig_three<0)
            recon_filt_two=np.where(eig_two<0)
            recon_filt_three=np.where(eig_one>=0)
        if (i=='cluster'):
            recon_filt_one=np.where(eig_three<0)
            recon_filt_two=np.where(eig_two<0)
            recon_filt_three=np.where(eig_one<0)
                
        recon_img[recon_filt_one]=1
        recon_img[recon_filt_two]=recon_img[recon_filt_two]+1
        recon_img[recon_filt_three]=recon_img[recon_filt_three]+1  

        recon_img=recon_img.flatten()
        recon_img=recon_img.astype(np.int8)#Just done to reduce memory
        mask=(recon_img !=3)
        mask_true=(recon_img ==3)
        del recon_img
                
        #Find and store relevent eigpairs
        if (i=='void'):
            mask_fnl[mask_true]=0#Identify final mask void-0
            
        if (i=='sheet'):
            vecsvals[mask,:,:]=np.ones((3,4))*9#wipe out all pixels which are not sheet

            fnd_prs=np.where(vecsvals[:,:,0]<0)#Find where eigenvectors which are perpendicular to sheet plane
            eig_fnl[fnd_prs[0],:]=vecsvals[fnd_prs[0],fnd_prs[1],:]#save relevent eigpairs into final output array
            mask_fnl[mask_true]=1#Identify final mask sheet-1

        if (i=='filament'):
            vecsvals[mask,:,:]=np.ones((3,4))*-9#wipe out all pixels which are not filament

            fnd_prs=np.where(vecsvals[:,:,0]>=0)#Find e3 eigenvectors using eigenvalues
            eig_fnl[fnd_prs[0],:]=vecsvals[fnd_prs[0],fnd_prs[1],:]#save relevent eigpairs into final output array
            mask_fnl[mask_true]=2#Identify final mask filament-2
            
        if (i=='cluster'):
            mask_fnl[mask_true]=3#Identify final mask cluster-3
            
    return eig_fnl,mask_fnl  
eig_fnl,mask_fnl= lss_classifier(lss,eigvals_unsorted,values,eig_one,eig_two,eig_three)#Function run
mask=np.reshape(mask_fnl,(grid_nodes,grid_nodes,grid_nodes))#Reshape mask_fnl
#END SECTION-----------------------------------------------------------------------------------------------------

# **IGNORE** SECTION 4.1 Plot scatter,smoothed, eigvals, classification mask,color scatter, table with halo-classif ratios----------------------------
classify_mask(mask,grid_nodes,slc,smooth_scl,plane)
colorscatter_plt(data,mask,grid_nodes,slc,smooth_scl,plane,sim_sz,plane_thickness,Mass_res,particles_filt,Xc_mult,Xc_minus,Yc_mult,Yc_minus,Zc_mult,Zc_minus)
#-----------------------------------------------------------------------------------

# SECTION 5. Take Dot Product of Spin-LSS------------------------------------------------------------------------
recon_vecs_flt_unnorm=eig_fnl[:,1:4]#take eigenvectors from eig_fnl
recon_vecs_flt_norm=skl.normalize(recon_vecs_flt_unnorm)#normalize eigenvectors to make sure.
recon_vecs=np.reshape(recon_vecs_flt_norm,(grid_nodes,grid_nodes,grid_nodes,3))#Reshape eigenvectors

#'data' format reminder: (Pos)XYZ(Mpc/h), (Vel)VxVyVz(km/s), (Ang. Mom)JxJyJz((Msun/h)*(Mpc/h)*km/s), (Vir. Mass)Mvir(Msun/h) & (Vir. Rad)Rvir(kpc/h)
partcl_halo_flt=np.where((data[:,9]/(Mass_res))>=particles_filt)#filter for halos with <N particles
data=data[partcl_halo_flt]#Filter out halos with <N particles
halo_mass=data[:,9]
log_halo_mass=np.log10(halo_mass)#convert into log10(M)
mass_intvl=(np.max(log_halo_mass)-np.min(log_halo_mass))/tot_mass_bins#log_mass value used to find mass interval

results=np.zeros((tot_mass_bins,4))# [Mass_min, Mass_max, Value, Error] 
for mass_bin in range(tot_mass_bins):
    
    #Positions
    Xc=data[:,0]#I must redefine these at each iteration since I am filtering them for each mass interval
    Yc=data[:,1]
    Zc=data[:,2]
    #Angular momenta
    Lx=data[:,6]
    Ly=data[:,7]
    Lz=data[:,8]

    low_int_mass=np.min(log_halo_mass)+mass_intvl*mass_bin#Calculate mass interval
    hi_int_mass=low_int_mass+mass_intvl#Calculate mass interval
    results[mass_bin,0]=low_int_mass#Store mass interval
    results[mass_bin,1]=hi_int_mass#Store mass interval
   
    #Create mask to filter out halos within mass interval
    mass_mask=np.zeros(len(log_halo_mass))
    loint=np.where(log_halo_mass>=low_int_mass)
    hiint=np.where(log_halo_mass<hi_int_mass)
    mass_mask[loint]=1
    mass_mask[hiint]=mass_mask[hiint]+1
    mass_indx=np.where(mass_mask==2)
    
    #Positions & Angular Momenta filtered for mass interval
    Xc=Xc[mass_indx]
    Yc=Yc[mass_indx]
    Zc=Zc[mass_indx]
    halos_mom=np.column_stack((Lx,Ly,Lz))
    halos_mom=halos_mom[mass_indx]
    
    norm_halos_mom=skl.normalize(halos_mom)#Normalize Angular Momenta
    halos=np.column_stack((Xc,Yc,Zc,norm_halos_mom))
       
    store_spin=[]#Store Dot Product of spin-LSS
    for i in range(len(Xc)):
       #Create index from halo coordinates
        grid_index_x=mth.trunc(halos[i,0]*Xc_mult-Xc_minus)      
        grid_index_y=mth.trunc(halos[i,1]*Yc_mult-Yc_minus) 
        grid_index_z=mth.trunc(halos[i,2]*Zc_mult-Zc_minus) 
        
        if (mask[grid_index_x,grid_index_y,grid_index_z]==2):#Calculate Dot Product for filament-2
            spin_dot=np.inner(halos[i,3:6],recon_vecs[grid_index_x,grid_index_y,grid_index_z,:]) 
            store_spin.append(spin_dot)

    store_spin=np.asarray(store_spin) 
    costheta=abs(store_spin)#Take absolute value as only Alignment counts.
          
    results[mass_bin,2]=np.mean(costheta)#Alignment value calc. and store
    
    #Calculating error using bootstrap resampling
    runs=200+mass_bin*300
    a=np.random.randint(low=0,high=len(costheta),size=(runs,len(costheta)))
    mean_set=np.mean(costheta[a],axis=1)
    results[mass_bin,3]=np.std(mean_set)#Store 1sigma error
    
print(results)    
#END SECTION & CODE---------------------------------------------------------------------------------------------------

# **IGNORE** SECTION 5.1 Plot correlation----------------------------
alignment_plt(grid_nodes,results)
vector_scatter(data,mask,recon_vecs,grid_nodes,sim_sz,particles_filt,slc,Xc_mult,Xc_minus,Yc_mult,Yc_minus,Zc_mult,Zc_minus,plane_thickness)
#-----------------------------------------------------------------------------------
