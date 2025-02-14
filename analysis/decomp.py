import xarray as xr
import matplotlib.pyplot as plt
import glob
import re
import math
from pathlib import Path
import xAnimate
import cmocean.cm as cm
import numpy as np



def dsFromPath(folderpaths):

    file_pattern = re.compile(r'.*?(\d+).*?')
    def get_order(file):
        match = file_pattern.match(Path(file).name)
        if not match:
            return math.inf
        return int(match.groups()[0])

    files = []
    for f in folderpaths:
        print(f)
        for file in list(glob.glob(f+"output_0*.nc")):
            files.append(file)

    files = sorted(files,key=get_order)
    print(files)
    datasets = []
    for i in files:
        datasets.append(xr.open_dataset(i))
    return xr.combine_nested(datasets,concat_dim="time")

folderpaths = ['/home/garrett/Projects/laddie/output/ref_2025-02-13_x14/']
ds = dsFromPath(folderpaths)
f = -1.37e-4
Cd = 2.5e-3
t0=10
u2 = ds.U2u[t0:].mean(dim="time").values
v2 = ds.V2v[t0:].mean(dim="time").values
dsbed = xr.open_dataset("../input/mitgcm.nc")
bed = dsbed.bedrockTopography.values
dx = (ds.x.values[1]-ds.x.values[0])
dy = (ds.y.values[1]-ds.y.values[0])

#h2 = 0.003*(ds.D2[t0:].mean(dim="time").values+bed)+ds.RL[t0:].mean(dim="time").values# + ds.rl[75:].mean(dim="time").values
h2 = ds.D2[t0:].mean(dim="time").values
#M = ds.RL[t0:].mean(dim="time").values*9.8+ds.TWtermav[t0:].mean(dim="time")#0.0003*(ds.D2[t0:].mean(dim="time").values+bed)
M = ds.RL[t0:].mean(dim="time").values+ds.TWtermav[t0:].mean(dim="time")#0.0003*(ds.D2[t0:].mean(dim="time").values+bed)
M = 0*ds.RL[t0:].mean(dim="time").values+0*ds.TWtermav[t0:].mean(dim="time")#0.0003*(ds.D2[t0:].mean(dim="time").values+bed)
M = (ds.RL[t0:].mean(dim="time").values + ds.TWtermav[t0:].mean(dim="time").values)


################################3
#fig, (ax1,ax2) = plt.subplots(1,2)
#c = ax1.imshow(ds.RL[t0:].mean(dim="time").values)
#plt.colorbar(c, ax=ax1)
#c = ax2.imshow(ds.TWtermav[t0:].mean(dim="time").values)
#plt.colorbar(c, ax=ax2)
#plt.show()
#################################3
#fig, (ax1,ax2) = plt.subplots(1,2)
#
#dMdx = (-M+np.roll(M,-1,axis=1))/dx
#dMdy = (-M+np.roll(M,-1,axis=0))/dy
#
#c = ax1.imshow(dMdx,vmin=-0.001,vmax=0.001,cmap="RdBu")
#plt.colorbar(c, ax=ax1)
#c = ax2.imshow(dMdy,vmin=-0.001,vmax=0.001,cmap="RdBu")
##plt.colorbar(c, ax=ax2)
#plt.show()




#M = 0.03*(ds.D2[t0:].mean(dim="time").values+bed)
breakpoint()
fig, (ax1,ax2,ax3) = plt.subplots(1,3)

dMdx = (-M+np.roll(M,-1,axis=1))/dx
dMdy = (-M+np.roll(M,-1,axis=0))/dy

honu = (h2+np.roll(h2,-1,axis=1))/2

honv = (h2+np.roll(h2,-1,axis=0))/2

intface = -((-dMdx*honu) + np.roll((dMdx*honu),-1,axis=0))/dy + ((-dMdy*honv) + np.roll((dMdy*honv),-1,axis=1))/dx
#intface = dMdx+dMdy

#dMdx = (dMdx+np.roll(dMdx,-1,axis=1))/2
#dMdy = (dMdy+np.roll(dMdy,-1,axis=0))/2

#dh2dx = (h2-np.roll(h2,1,axis=1))/dx
#dh2dy = (h2-np.roll(h2,1,axis=0))/dy

#dh2dx = (dh2dx+np.roll(dh2dx,-1,axis=1))/2
#dh2dy = (dh2dy+np.roll(dh2dy,-1,axis=0))/2

dhdx = (-h2+np.roll(h2,-1,axis=1))/dx
dhdy = (-h2+np.roll(h2,-1,axis=0))/dy

u2mag = np.sqrt(u2**2+v2**2)
curl = (-u2*u2mag+np.roll(u2*u2mag,-1,axis=0))/dy
curl2 = (-v2*u2mag+np.roll(v2*u2mag,-1,axis=1))/dx
curl = -curl+curl2
X,Y = np.meshgrid(ds.x.values,ds.y.values)
ax1.pcolormesh(X,Y,f*(ds.ent2+ds.entr-ds.detr)[t0:].mean(dim="time")/(60*60*24*365),vmin=-8e-9,vmax=8e-9,cmap="RdBu_r")
#ax2.pcolormesh(X,Y,intface,vmin=-8e-7,vmax=8e-7,cmap="RdBu_r")
ax2.pcolormesh(X,Y,(u2*dhdx + v2*dhdy)*(f**1),vmin=-8e-9,vmax=8e-9,cmap="RdBu_r")
ax3.pcolormesh(X,Y,Cd*curl,vmin=-8e-9,vmax=8e-9,cmap="RdBu_r")
plt.show()
#ax3.imshow(Cd*ds.U2[10:].mean(dim="time"))



