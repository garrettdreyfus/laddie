#fields needed
#floatingmask
from scipy.io import loadmat
from netCDF4 import Dataset
import numpy as np
import matplotlib.pyplot as plt


variables = loadmat("/home/garrett/Projects/MITgcm_ISC/experiments/widthexp-GLIB-explore-32/at125w250/input/metaparameters.mat",struct_as_record=False)
xx = np.asarray(variables["xx"]).flatten()
yy = np.asarray(variables["yy"]).flatten()

lowerSurface = np.asarray(variables["icedraft"])
bedrockTopography = np.asarray(variables["h"])
groundedMask = np.asarray((variables["h"]==variables["icedraft"]))
#floatingMask = np.asarray(np.logical_and(variables["h"]!=variables["icedraft"],variables["icedraft"]<0))
floatingMask=~groundedMask
lowerSurface[np.logical_and(lowerSurface>-250,floatingMask)] = -250
plt.imshow(lowerSurface)
plt.show()

ncfile = Dataset('input/mitgcm_250w_allice.nc',mode='w',format='NETCDF4_CLASSIC') 

y_dim = ncfile.createDimension('y', yy.shape[0])     # latitude axis
x_dim = ncfile.createDimension('x', xx.shape[0])

x = ncfile.createVariable('x', np.float32, ('x',))
x[:] = xx
y = ncfile.createVariable('y', np.float32, ('y',))
y[:] = yy

ls = ncfile.createVariable('lowerSurface',np.float64,('y','x'))
ls[:] = lowerSurface.T
print(ls.shape)


brt = ncfile.createVariable('bedrockTopography',np.float64,('y','x'))
brt[:] = bedrockTopography.T
print(brt.shape)
plt.imshow(bedrockTopography)
plt.show()


gm = ncfile.createVariable('groundedMask',np.float64,('y','x'))
gm[:] = groundedMask.T
print(gm.shape)

fm = ncfile.createVariable('floatingMask',np.float64,('y','x'))
fm[:] = floatingMask.T
print(fm.shape)

ncfile.close()
