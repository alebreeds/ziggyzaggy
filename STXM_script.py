import tifffile
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy import optimize
from scipy.interpolate import interp1d

raw_1 = tifffile.imread(r"C:\...") # TIFF stack
energies = np.loadtxt(r"C:\...") # one-column energy list
sample_name = '90b'

# Initialize the new array with the same shape
baseline = np.zeros_like(raw_1)

# Iterate over the second and third axes
for i in range(raw_1.shape[1]):  # axis=1
    for j in range(raw_1.shape[2]):  # axis=2
        start_value = raw_1[0, i, j]
        end_value = raw_1[-1, i, j]
        baseline[:, i, j] = np.linspace(start_value, end_value, raw_1.shape[0])
        
raw = raw_1 - baseline


# Defining the functions

def normalize(tif,pre,cutoff):
    '''
    Given a stack (numpy array, energies along 0th axis), the function return a normalised stack in which the minimum is the mean value 
    of the intensities before the "pre" level, and the maximum is extracted in the levels until the "cutoff" level.
    '''
    min_values = np.mean(tif[0:pre,:,:], axis=0, keepdims=True)
    max_values = np.max(tif[0:cutoff,:,:], axis=0, keepdims=True)
    normalized_array = (tif-min_values)/(max_values-min_values)
    normalized_array[0:pre,:,:] = 0
    return np.where(normalized_array > 0, normalized_array, 0)               

def filter_stack(tif, filter):
    zs, ys, xs = filter.shape
    a = tif
    for i in range(zs):
        for j in range(ys):
            for k in range(xs):
                if filter[i,j,k] == True:
                    a[i,j,k] = tif[i,j,k]
                else:
                    a[i,j,k] = 0
    return a

def divide(a,b):
    c = np.where(b == 0, 100000, b)
    return np.divide(a, c)
   
def std_filtered(a,filter):
    c = np.zeros_like(a)
    avg = np.sum(a)/np.sum(filter)
    zs, ys = filter.shape
    for i in range(zs):
        for j in range(ys):
            if filter[i,j] == True:
                c[i,j] = ((a[i,j]-avg)**2)
            else:
                c[i,j] = 0
    return (np.sum(c)/(np.sum(filter)-1))**0.5

def covariance(a,b,filter):
    c = np.zeros_like(a)
    zs, ys = filter.shape
    for i in range(zs):
        for j in range(ys):
            if filter[i,j] == True:
                c[i,j] = a[i,j]/(np.sum(a)/np.sum(filter))*(b[i,j]/(np.sum(b)/np.sum(filter)))
            else:
                c[i,j] = 0
    return np.sum(c)/(np.sum(filter)-1)

# Making and saving the tif of the normalised stack

norm = normalize(raw,10,35)
tifffile.imwrite(sample_name+'_crop_norm.tiff', norm, photometric='minisblack') 
plt.imshow(np.average(raw, axis=0))
plt.show()

# Creating a filter array to exclude background

filterarr_tot = (np.average(raw, axis=0) > 0.03) & (np.average(raw, axis=0) < 0.09)
filterarr = np.stack((filterarr_tot,filterarr_tot,filterarr_tot), axis=0)
int_filterarr = filter_stack(np.ones_like(filterarr),filterarr)
true_number = np.count_nonzero(filterarr_tot)
whole_filter = np.tile(filterarr_tot, (raw.shape[0],1,1))

# In this case I can recognize the energy corresponding to Mn2, 3 and 4 in the 14th, 17th and 19th slice respectively (index 13, 16, 18)

mn2 = norm[17,:,:] - np.min(norm[17,:,:], where=filterarr_tot, initial = 20)
mn3 = norm[20,:,:]
mn4 = norm[24,:,:]

val_int_matrix = np.stack((mn2,mn3,mn4), axis=0)
filtered_valintmatrix = filter_stack(val_int_matrix, filterarr)

# Color map creation

cmap_background = plt.cm.colors.ListedColormap(['black', 'none'])

# Species vs Mn(II)

val_int_matrix_norm_vsmn2 = filter_stack(divide(val_int_matrix, np.stack((mn2,mn2,mn2), axis=0)), filterarr)

fig, axs = plt.subplots(1, 2, figsize=(12, 4))
fig1 = axs[0].imshow(val_int_matrix_norm_vsmn2[1,:,:], cmap='jet', vmin=0, vmax=2.5)
axs[0].set_title('Mn(III)/Mn(II)')
plt.colorbar(mappable=fig1, ax=axs[0])
fig1 = axs[0].imshow(int_filterarr[0,:,:], cmap=cmap_background)
fig2 = axs[1].imshow(val_int_matrix_norm_vsmn2[2,:,:], cmap='jet', vmin=0, vmax=2.5)
axs[1].set_title('Mn(IV)/Mn(II)')
plt.colorbar(mappable=fig2, ax=axs[1])
fig2 = axs[1].imshow(int_filterarr[0,:,:], cmap=cmap_background)
plt.savefig(sample_name+'_vsMnII.png')
plt.tight_layout()
plt.show()

# Species vs Mn(III)

val_int_matrix_norm_vsmn3 = filter_stack(divide(val_int_matrix, np.stack((mn3,mn3,mn3), axis=0)), filterarr)

fig, axs = plt.subplots(1, 2, figsize=(12, 4))
fig1 = axs[0].imshow(val_int_matrix_norm_vsmn3[0,:,:], cmap='jet', vmin=0, vmax=2.5)
axs[0].set_title('Mn(II)/Mn(III)')
plt.colorbar(mappable=fig1, ax=axs[0])
fig1 = axs[0].imshow(int_filterarr[0,:,:], cmap=cmap_background)
fig2 = axs[1].imshow(val_int_matrix_norm_vsmn3[2,:,:], cmap='jet', vmin=0, vmax=2.5)
axs[1].set_title('Mn(IV)/Mn(III)')
plt.colorbar(mappable=fig2, ax=axs[1])
fig2 = axs[1].imshow(int_filterarr[0,:,:], cmap=cmap_background)
plt.savefig(sample_name+'_vsMnIII.png')
plt.tight_layout()
plt.show()

# Species vs Mn(IV)

val_int_matrix_norm_vsmn4 = filter_stack(divide(val_int_matrix, np.stack((mn4,mn4,mn4), axis=0)), filterarr)

fig, axs = plt.subplots(1, 2, figsize=(12, 4))
fig1 = axs[0].imshow(val_int_matrix_norm_vsmn4[0,:,:], cmap='jet', vmin=0, vmax=2.5)
axs[0].set_title('Mn(II)/Mn(IV)')
plt.colorbar(mappable=fig1, ax=axs[0])
fig1 = axs[0].imshow(int_filterarr[0,:,:], cmap=cmap_background)
fig2 = axs[1].imshow(val_int_matrix_norm_vsmn4[1,:,:], cmap='jet', vmin=0, vmax=2.5)
axs[1].set_title('Mn(III)/Mn(IV)')
plt.colorbar(mappable=fig2, ax=axs[1])
fig2 = axs[1].imshow(int_filterarr[0,:,:], cmap=cmap_background)
plt.savefig(sample_name+'_vsMnIV.png')
plt.tight_layout()
plt.show()


mn_tot = np.add(mn2,np.add(mn3,mn4))
mn2_norm = np.divide(mn2,mn_tot)
mn3_norm = np.divide(mn3,mn_tot)
mn4_norm = np.divide(mn4,mn_tot)
mn_norm_stack_val = np.stack((mn2_norm, mn3_norm, mn4_norm), axis=0)
mn_norm_stack_val_filt = filter_stack(mn_norm_stack_val, filterarr)

# Mn2
mn2_avg = np.nansum(mn_norm_stack_val_filt[0,:,:])/true_number
mn2_avg_matrix = mn2_avg*int_filterarr[0,:,:]
mn2_var = np.nansum(((mn_norm_stack_val_filt[0,:,:]-mn2_avg_matrix)**2))/(true_number-1)
mn2_std = mn2_var**0.5

# Mn3
mn3_avg = np.nansum(mn_norm_stack_val_filt[1,:,:])/true_number
mn3_avg_matrix = mn3_avg*int_filterarr[0,:,:]
mn3_var = np.nansum(((mn_norm_stack_val_filt[1,:,:]-mn3_avg_matrix)**2))/(true_number-1)
mn3_std = mn3_var**0.5

# Mn4
mn4_avg = np.nansum(mn_norm_stack_val_filt[2,:,:])/true_number
mn4_avg_matrix = mn4_avg*int_filterarr[0,:,:]
mn4_var = np.nansum(((mn_norm_stack_val_filt[2,:,:]-mn4_avg_matrix)**2))/(true_number-1)
mn4_std = mn4_var**0.5


fig, axs = plt.subplots(1, 3, figsize=(14, 4))
fig1 = axs[0].imshow(mn_norm_stack_val_filt[0,:,:], cmap='jet')
axs[0].set_title('Mn(II) norm')
plt.colorbar(mappable=fig1, ax=axs[0])
fig1 = axs[0].imshow(int_filterarr[0,:,:], cmap=cmap_background)
fig2 = axs[1].imshow(mn_norm_stack_val_filt[1,:,:], cmap='jet')
axs[1].set_title('Mn(III) norm')
plt.colorbar(mappable=fig2, ax=axs[1])
fig2 = axs[1].imshow(int_filterarr[0,:,:], cmap=cmap_background)
fig3 = axs[2].imshow(mn_norm_stack_val_filt[2,:,:], cmap='jet')
axs[2].set_title('Mn(IV) norm')
plt.colorbar(mappable=fig3, ax=axs[2])
fig3 = axs[2].imshow(int_filterarr[0,:,:], cmap=cmap_background)
plt.suptitle(sample_name+ ' Valence distribution')
plt.savefig(sample_name+'_val_map.png')
plt.tight_layout()
plt.show()


fig, axs = plt.subplots(1, 3, figsize=(14, 4))
fig1 = axs[0].imshow(mn_norm_stack_val_filt[0,:,:], cmap='jet', vmin=0, vmax=1)
axs[0].set_title('Mn(II) norm')
plt.colorbar(mappable=fig1, ax=axs[0])
fig1 = axs[0].imshow(int_filterarr[0,:,:], cmap=cmap_background)
fig2 = axs[1].imshow(mn_norm_stack_val_filt[1,:,:], cmap='jet', vmin=0, vmax=1)
axs[1].set_title('Mn(III) norm')
plt.colorbar(mappable=fig2, ax=axs[1])
fig2 = axs[1].imshow(int_filterarr[0,:,:], cmap=cmap_background)
fig3 = axs[2].imshow(mn_norm_stack_val_filt[2,:,:], cmap='jet', vmin=0, vmax=1)
axs[2].set_title('Mn(IV) norm')
plt.colorbar(mappable=fig3, ax=axs[2])
fig3 = axs[2].imshow(int_filterarr[0,:,:], cmap=cmap_background)
plt.suptitle(sample_name+ ' Valence distribution (rescaled)')
plt.savefig(sample_name+'_val_map_rescaled.png')
plt.tight_layout()
plt.show()

plt.imshow(mn_tot, cmap='jet')
plt.colorbar()
plt.imshow(int_filterarr[0,:,:], cmap=cmap_background)
plt.title(sample_name+' total Mn distribution')
plt.savefig(sample_name+'_total_mn.png')
plt.show()

# Covariance

mn2varred = np.divide(mn_norm_stack_val_filt[0,:,:],mn2_avg)-1*filterarr_tot
mn3varred = np.divide(mn_norm_stack_val_filt[1,:,:],mn3_avg)-1*filterarr_tot
mn4varred = np.divide(mn_norm_stack_val_filt[2,:,:],mn4_avg)-1*filterarr_tot
mn2stdred = np.nansum((np.divide(mn_norm_stack_val_filt[0,:,:],mn2_avg)-1*filterarr_tot)**2)
mn3stdred = np.nansum((np.divide(mn_norm_stack_val_filt[1,:,:],mn3_avg)-1*filterarr_tot)**2)
mn4stdred = np.nansum((np.divide(mn_norm_stack_val_filt[2,:,:],mn4_avg)-1*filterarr_tot)**2)

mn2mn3_cov = (np.nansum(mn2varred*mn3varred))/((mn2stdred*mn3stdred)**0.5)*np.sign(mn2_avg*mn3_avg)
mn2mn4_cov = (np.nansum(mn2varred*mn4varred))/((mn2stdred*mn4stdred)**0.5)*np.sign(mn2_avg*mn4_avg)
mn3mn4_cov = (np.nansum(mn3varred*mn4varred))/((mn3stdred*mn4stdred)**0.5)*np.sign(mn3_avg*mn4_avg)

print(mn2_avg, mn2_std)
print(mn3_avg, mn3_std)
print(mn4_avg, mn4_std)
print(mn2mn3_cov, mn2mn4_cov, mn3mn4_cov)

average_summation_vectors = filter_stack(norm, whole_filter)
average_spectrum = np.sum(average_summation_vectors, axis = (1,2))/true_number

plt.plot(energies, average_spectrum)
plt.savefig(sample_name+'_average_spectrum.png')
plt.show()

export_average_spec = np.stack((energies,average_spectrum), axis=1)
np.savetxt('avg_spectrum.txt', export_average_spec)


