import numpy as np
import h5py
import sys
sys.path.insert(0, '/import/oth3/ajib0457/wang_peng_code/CIC_LSS/py')
import idlsave
import matplotlib.pyplot as plt
from matplotlib import cm
import matplotlib.colors as mcolors

grid_nodes=128
#its wise to compare the eigenpairs first, then if they don't agree, go back to hessian.
s=idlsave.read('/import/oth3/ajib0457/wang_peng_code/CIC_LSS/LSS_SAV/CIC_star_135_1282.0_Hessian_Eigen_cell.sav')
a=s.hessian_eigen_cell

eigvals=np.asarray(a['eigenvalues'])
eigvals_peng=np.array(list(eigvals), dtype=np.float)
eigvals_peng=np.reshape(eigvals_peng.transpose(),(grid_nodes**3,3))

eigvecs=np.asarray(a['eigenvectors'])
eigvecs_peng=np.array(list(eigvecs), dtype=np.float)
eigvecs_peng=np.reshape(eigvecs_peng,(9,grid_nodes,grid_nodes,grid_nodes))
eigvecs_peng=np.reshape(eigvecs_peng.transpose(),(grid_nodes**3,3,3))

f=h5py.File("/import/oth3/ajib0457/Peng_test_data_run/correl/my_den/files/output_files/oth3_eigpairs/peng_sample_eigpairs.h5" , 'r')
eigvecs_my=f['/eigvecs'][:]
eigvals_my=f['/eigvals'][:]
f.close()

eigvals_my=np.reshape(eigvals_my,(grid_nodes**3,3))
eigvecs_my=np.reshape(eigvecs_my,(grid_nodes**3,3,3))

#print(sum(sum(abs(eigvals_my)-abs(eigvals_peng))))
#print(sum(sum(sum(abs(eigvecs_my)-abs(eigvecs_peng)))))

lss=['filament','void','sheet','cluster']#Choose which LSS you would like to get classified
def lss_classifier(lss,eigvals_unsorted,eigvecs):
    
    ####Classifier#### 
    '''
    This is the classifier function which takes input:
    
    lss: the labels of Large scale structure which will be identified pixel by pixel and also eigenvectors 
    will be retrieved if applicable.
    
    vecsvals: These are the eigenvalues and eigevector pairs which correspond row by row.
    
    eig_one,two and three: These are the isolated eigenvalues 
    
    This function will output:
    
    eig_fnl: An array containing all of the relevent eigenvectors for each LSS type
    
    mask_fnl: array prescribing 0-void, 1-sheet, 2-filament and 3-cluster
    
    '''
    eigvals=np.sort(eigvals_unsorted) 
    eig_one=eigvals[:,2]
    eig_two=eigvals[:,1]
    eig_three=eigvals[:,0]
    #link eigenvalues as keys to eigenvectors as values inside dictionary    
    vec_arr_num,vec_row,vec_col=np.shape(eigvecs)
    values=np.reshape(eigvecs.transpose(0,2,1),(vec_row*vec_arr_num,vec_col))#orient eigenvectors so that each row is an eigenvector
    
    eigvals_unsorted=eigvals_unsorted.flatten()
    vecsvals=np.column_stack((eigvals_unsorted,values))
    
    eig_fnl=np.zeros((grid_nodes**3,4))
    mask_fnl=np.zeros((grid_nodes**3))
    for i in lss:
        vecsvals=np.column_stack((eigvals_unsorted,values))
        recon_img=np.zeros([grid_nodes**3])
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
        
        #LSS FILTER#
        recon_img[recon_filt_one]=1
        recon_img[recon_filt_two]=recon_img[recon_filt_two]+1
        recon_img[recon_filt_three]=recon_img[recon_filt_three]+1  
        del recon_filt_one
        del recon_filt_two
        del recon_filt_three
        recon_img=recon_img.flatten()
        recon_img=recon_img.astype(np.int8)
        mask=(recon_img !=3)#Up to this point, a mask is created to identify where there are NO filaments...
        mask_true=(recon_img ==3)
        del recon_img
        vecsvals=np.reshape(vecsvals,(grid_nodes**3,3,4))
                
        #Find relevent eigpairs
        if (i=='void'):#There is no appropriate axis of a void?
            mask_fnl[mask_true]=0
            del mask_true
            
        if (i=='sheet'):
            vecsvals[mask,:,:]=np.ones((3,4))*9#...which are then converted into -9 at this point
            del mask
            fnd_prs=np.where(vecsvals[:,:,0]<0)#find LSS axis
            eig_fnl[fnd_prs[0],:]=vecsvals[fnd_prs[0],fnd_prs[1],:]
            mask_fnl[mask_true]=1
            del mask_true

        if (i=='filament'):
            vecsvals[mask,:,:]=np.ones((3,4))*-9#...which are then converted into -9 at this point
            del mask
            fnd_prs=np.where(vecsvals[:,:,0]>=0)#find LSS axis
            eig_fnl[fnd_prs[0],:]=vecsvals[fnd_prs[0],fnd_prs[1],:]
            mask_fnl[mask_true]=2
            del mask_true
            
        if (i=='cluster'):#There is no appropriate axis of a void?
            mask_fnl[mask_true]=3
            del mask_true            
        
    return eig_fnl,mask_fnl  
    
eig_fnl_my,mask_fnl_my= lss_classifier(lss,eigvals_my,eigvecs_my)#Function run
eig_fnl_peng,mask_fnl_peng= lss_classifier(lss,eigvals_peng,eigvecs_peng)#Function run
mask_fnl_my=np.reshape(mask_fnl_my,(grid_nodes,grid_nodes,grid_nodes))
mask_fnl_peng=np.reshape(mask_fnl_peng,(grid_nodes,grid_nodes,grid_nodes))

#Density field smoothed
f=h5py.File('/import/oth3/ajib0457/Peng_test_data_run/my_den/den_grid%s_halo_bin_wang_peng_stars'%grid_nodes, 'r')
image=f['/stars'][:]
f.close()
image=np.reshape(image,(grid_nodes,grid_nodes,grid_nodes))
s=3.92
in_val,fnl_val=-140,140
X,Y,Z=np.meshgrid(np.linspace(in_val,fnl_val,grid_nodes),np.linspace(in_val,fnl_val,grid_nodes),np.linspace(in_val,fnl_val,grid_nodes))
h=(1/np.sqrt(2*np.pi*s*s))**(3)*np.exp(-1/(2*s*s)*(Y**2+X**2+Z**2))
h=np.roll(h,int(grid_nodes/2),axis=0)
h=np.roll(h,int(grid_nodes/2),axis=1)
h=np.roll(h,int(grid_nodes/2),axis=2)
fft_dxx=np.fft.fftn(h)
fft_db=np.fft.fftn(image)
ifft_a=np.fft.ifftn(np.multiply(fft_dxx,fft_db)).real


slc=100

plt.figure(figsize=(15,15),dpi=100)
#The two function below are purely for the color scheme of the imshow plot: Classifier, used to create discrete imshow
def colorbar_index(ncolors, cmap):
    cmap = cmap_discretize(cmap, ncolors)
    mappable = cm.ScalarMappable(cmap=cmap)
    mappable.set_array([])
    mappable.set_clim(-0.5, ncolors+0.5)
    colorbar = plt.colorbar(mappable)
    colorbar.set_ticks(np.linspace(0, ncolors, ncolors))
    colorbar.set_ticklabels(range(ncolors))

def cmap_discretize(cmap, N):   
    if type(cmap) == str:
        cmap = plt.get_cmap(cmap)
    colors_i = np.concatenate((np.linspace(0, 1., N), (0.,0.,0.,0.)))
    colors_rgba = cmap(colors_i)
    indices = np.linspace(0, 1., N+1)
    cdict = {}
    for ki,key in enumerate(('red','green','blue')):
        cdict[key] = [ (indices[i], colors_rgba[i-1,ki], colors_rgba[i,ki])
                       for i in xrange(N+1) ]
    # Return colormap object.
    return mcolors.LinearSegmentedColormap(cmap.name + "_%d"%N, cdict, 1024)

#Classifier: This subplot must be first so that the two functions above will help to discretise the color scheme and color bar
#Peng data
ax=plt.subplot2grid((2,2), (1,0))  
plt.title('Classifier Peng')
#plt.xlabel('z')
#plt.ylabel('x')
cmap = plt.get_cmap('jet')#This is where you can change the color scheme
ax.imshow(np.rot90(mask_fnl_peng[:,slc,:],1), interpolation='nearest', cmap=cmap,extent=[0,128,0,128])#The colorbar will adapt to data
colorbar_index(ncolors=4, cmap=cmap)

#my data
ax=plt.subplot2grid((2,2), (1,1))  
plt.title('Classifier')
#plt.xlabel('z')
#plt.ylabel('x')
cmap = plt.get_cmap('jet')#This is where you can change the color scheme
ax.imshow(np.rot90(mask_fnl_my[:,slc,:],1), interpolation='nearest', cmap=cmap,extent=[0,128,0,128])#The colorbar will adapt to data
colorbar_index(ncolors=4, cmap=cmap)

#Density field my data
ax5=plt.subplot2grid((2,2), (0,1))    
plt.title('density field')
cmapp = plt.get_cmap('jet')
scl_plt=35#reduce scale of density fields and eigenvalue subplots by increasing number
dn_fl_plt=ax5.imshow(np.power(np.rot90(image[:,slc,:],1),1./scl_plt),cmap=cmapp,extent=[0,128,0,128])#The colorbar will adapt to data
plt.colorbar(dn_fl_plt,cmap=cmapp)

#smoothed Density field my data
ax5=plt.subplot2grid((2,2), (0,0))    
plt.title('smoothed density field')
cmapp = plt.get_cmap('jet')
scl_plt=35#reduce scale of density fields and eigenvalue subplots by increasing number
dn_fl_plt=ax5.imshow(np.power(np.rot90(ifft_a[:,slc,:],1),1./scl_plt),cmap=cmapp,extent=[0,128,0,128])#The colorbar will adapt to data
plt.colorbar(dn_fl_plt,cmap=cmapp)

sim_sz=75#Mpc
#Calculate the std deviation in physical units
grid_phys=1.*sim_sz/grid_nodes#Size of each voxel in physical units
val_phys=1.*(2*fnl_val)/grid_nodes#Value in each grid voxel
std_dev_phys=1.*s/val_phys*grid_phys

plt.savefig('/import/oth3/ajib0457/Peng_test_data_run/plots/classification/grid%s_slc%s_smth%s.png' %(grid_nodes,slc,std_dev_phys))
