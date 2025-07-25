import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import glob
import re
import math
from pathlib import Path
from astropy.convolution import convolve, Box2DKernel
import xAnimate
import numpy
import ipdb
import cmocean


folderpaths = ['/home/garrett/Projects/laddie/output/ref_2025-07-08_smallerminD/']
# folderpaths = ['/home/garrett/Projects/laddie/output/ref_2025-07-09_no_mom_adv/']
# folderpaths = ['/home/garrett/Projects/laddie/output/ref_2025-07-09_no_mom_adv_no_melt/']

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
ds = xr.combine_nested(datasets,concat_dim="time")

def frame_func_v( data,t ):
    f,a = plt.subplots(1,1)
    data.plot.pcolormesh(vmin=-0.15,vmax=0.15,cmap="RdBu_r")
    f.suptitle(t)
    return f

def frame_func_d( data,t ):
    f,a = plt.subplots(1,1)
    data.plot.pcolormesh(vmin=1,vmax=100,cmap="magma")
    f.suptitle(t)
    return f

def frame_func_d2( data,t ):
    f,a = plt.subplots(1,1)
    data.plot.pcolormesh(vmin=10,vmax=1000,cmap="magma")
    f.suptitle(t)
    return f

def frame_func_T( data,t ):
    f,a = plt.subplots(1,1)
    data.plot.pcolormesh(vmin=-2,vmax=2,cmap="magma")
    f.suptitle(t)
    return f

def frame_func_S( data,t ):
    f,a = plt.subplots(1,1)
    data.plot.pcolormesh(vmin=34.2,vmax=35.7,cmap="magma")
    f.suptitle(t)
    return f

def frame_func_melt( data,t ):
    f,a = plt.subplots(1,1)
    data.plot.pcolormesh(cmap="RdBu_r")
    f.suptitle(t)
    return f

def frame_func_twterm( data,t  ):
    f,a = plt.subplots(1,1)
    data.plot.pcolormesh(vmin=-1000,vmax=200,cmap="magma")
    f.suptitle(t)
    return f

def frame_func_relvort( data, t ):
    f,a = plt.subplots(1,1)
    data.plot.pcolormesh(vmin=-8e-5,vmax=8e-5,cmap="RdBu_r")
    f.suptitle(t)
    return f

def frame_func_pv( data,t ):
    f,a = plt.subplots(1,1)
    data.plot.pcolormesh(vmin=-4e-6,vmax=4e-6,cmap="RdBu_r")
    f.suptitle(t)
    return f



anim = False
if anim:

    dx = (ds.x.values[1]-ds.x.values[0])
    dy = (ds.y.values[1]-ds.y.values[0])

    ds['relvort'] = (ds.T.dims,(np.roll(ds.V2v,-1,axis=1)-np.roll(ds.V2v,1,axis=1))/dx - (np.roll(ds.U2u,-1,axis=0)-np.roll(ds.U2u,1,axis=0))/dy)
    ds['pv'] = (ds.T.dims,(ds.relvort.data-1.37e-4)/ds.D2.data)

    xAnimate.make_animation( ds.U2t, fp_out = './U2.mp4',
                            anim_dim = 'time', fps=2,
                            frame_func = frame_func_v)

    xAnimate.make_animation( ds.relvort, fp_out = './RELVORT.mp4',
                             anim_dim = 'time', fps=2,
                             frame_func = frame_func_relvort)


    xAnimate.make_animation( ds.pv, fp_out = './PV.mp4',
                             anim_dim = 'time', fps=2,
                             frame_func = frame_func_pv)




    xAnimate.make_animation( ds.melt, fp_out = './melt.mp4',
                             anim_dim = 'time',fps=2,
                             frame_func = frame_func_melt)

    xAnimate.make_animation( ds.Ut, fp_out = './U.mp4',
                            anim_dim = 'time',fps=2,
                            frame_func = frame_func_v)
    ##
    #
    xAnimate.make_animation( ds.Vt, fp_out = './V.mp4',
                            anim_dim = 'time',fps=2,
                            frame_func = frame_func_v)
    ##
    xAnimate.make_animation( ds.V2t, fp_out = './V2.mp4',
                            anim_dim = 'time', fps=2,
                            frame_func = frame_func_v)
    #
    xAnimate.make_animation( ds.D, fp_out = './D.mp4',
                            anim_dim = 'time', fps=2,
                            frame_func = frame_func_d)
    ##
    xAnimate.make_animation( ds.D2, fp_out = './D2.mp4',
                             anim_dim = 'time', fps=2,
                             frame_func = frame_func_d2)

    xAnimate.make_animation( ds.S, fp_out = './S.mp4',
                             anim_dim = 'time', fps=2,
                             frame_func = frame_func_S)

    xAnimate.make_animation( ds.T, fp_out = './T.mp4',
                             anim_dim = 'time', fps=2,
                             frame_func = frame_func_T)

def moving_average(data, window_size):
    weights = np.ones(window_size) / window_size
    return np.convolve(data, weights, mode='valid')

#ds.melt.where(ds.y<150000).where(ds.time>20).mean(dim="x").mean(dim="y").plot()
#plt.show()
fig,ax1 = plt.subplots(1,1,figsize=(12,12))

X,Y = np.meshgrid(ds.x.values/1000,ds.y.values/1000)
starttime=150
U2 = ds.where(ds.time>starttime).U2t.mean(dim="time").values
V2 = ds.where(ds.time>starttime).V2t.mean(dim="time").values
# im = ax1.pcolormesh(X,Y,ds.D2.where(ds.time>starttime).mean(dim="time"),cmap=cmocean.cm.deep)
im = ax1.pcolormesh(X,Y,ds.where(ds.time>starttime).melt.mean(dim="time").where(ds.y<150000),vmin=0,vmax=60,cmap="Reds")
cbar = plt.colorbar(im,ax=ax1)
cbar.set_label("Melt (m/yr)",fontsize=16)
box_kernel = Box2DKernel(5,mode="center")

#X = convolve(X, box_kernel,preserve_nan=True)
#Y = convolve(Y, box_kernel,preserve_nan=True)
#
U2 = convolve(U2, box_kernel,preserve_nan=True)
V2 = convolve(V2, box_kernel,preserve_nan=True)

U2[np.logical_or(U2==0,V2==0)] = np.nan
V2[np.logical_or(U2==0,V2==0)] = np.nan
 
Q = ax1.quiver(X[::2,::2],Y[::2,::2],U2[::2,::2],V2[::2,::2],scale=0.4,width=0.0035,color="navy")

ax1.quiverkey(Q, 115, 40, 0.05,"0.05 (m/s)",angle = 90,coordinates="data")

ax1.set_xlabel("X (km)",fontsize=16)
ax1.set_ylabel("Y (km)",fontsize=16)
ax1.set_ylim(20,170)
ax1.set_xlim(100,300)

plt.show()

fig,ax2 = plt.subplots(1,1,figsize=(9,6))

U = ds.where(ds.time>starttime).Ut.mean(dim="time").values
V = ds.where(ds.time>starttime).Vt.mean(dim="time").values
im = ax2.pcolormesh(X,Y,ds.where(ds.time>starttime).melt.mean(dim="time").where(ds.y<150000),vmin=0,vmax=60,cmap="Reds")
cbar = plt.colorbar(im, ax=ax2)
cbar.set_label("(m/yr)",fontsize=16)
box_kernel = Box2DKernel(5,mode="center")

#X = convolve(X, box_kernel,preserve_nan=True)
#Y = convolve(Y, box_kernel,preserve_nan=True)
#
#U = convolve(U, box_kernel,preserve_nan=True)
#V = convolve(V, box_kernel,preserve_nan=True)

U[np.logical_or(U==0,V==0)] = np.nan
V[np.logical_or(U==0,V==0)] = np.nan
 
ax2.quiver(X[::2,::2],Y[::2,::2],U[::2,::2],V[::2,::2],scale=1,width=0.0035,color="navy")
ax2.set_xlabel("X (km)",fontsize=16)
ax2.set_ylabel("Y (km)",fontsize=16)
ax2.set_ylim(20,170)
plt.show()
