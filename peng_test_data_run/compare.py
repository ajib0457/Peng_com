import numpy as np
import h5py
import sys
sys.path.insert(0, '/import/oth3/ajib0457/wang_peng_code/CIC_LSS/py')
import idlsave
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

print(sum(sum(abs(eigvals_my)-abs(eigvals_peng))))
print(sum(sum(sum(abs(eigvecs_my)-abs(eigvecs_peng)))))

lss=['filament']#Choose which LSS you would like to get classified
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

