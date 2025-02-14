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


#folderpaths = ['/home/garrett/Projects/laddie/output/ref_2024-11-26_noavrestart/','/home/garrett/Projects/laddie/output/ref_2024-11-26_restart24/']
#folderpaths = ['/home/garrett/Projects/laddie/output/ref_2024-11-27_w100/']
#folderpaths = ['/home/garrett/Projects/laddie/output/ref_2024-11-28_zootopia/']
#folderpaths = ['/home/garrett/Projects/laddie/output/ref_2025-02-03_momentumentrain/']
folderpaths = ['/home/garrett/Projects/laddie/output/ref_2025-02-13_x14/']
#folderpaths = ['/home/garrett/Projects/laddie/output/ref_2024-12-02_tenthtau/']

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
ds.V2t[20:].mean(dim="time").plot.pcolormesh()
plt.show()


def frame_func_v( data ):
    f,a = plt.subplots(1,1)
    data.plot.pcolormesh(vmin=-0.2,vmax=0.2,cmap="RdBu_r")
    return f

def frame_func_d( data ):
    f,a = plt.subplots(1,1)
    data.plot.pcolormesh(vmin=10,vmax=100,cmap="magma")
    return f

def frame_func_d2( data ):
    f,a = plt.subplots(1,1)
    data.plot.pcolormesh(vmin=10,vmax=1000,cmap="magma")
    return f

def frame_func_T( data ):
    f,a = plt.subplots(1,1)
    data.plot.pcolormesh(vmin=-2,vmax=2,cmap="magma")
    return f

def frame_func_S( data ):
    f,a = plt.subplots(1,1)
    data.plot.pcolormesh(vmin=34.2,vmax=35.7,cmap="magma")
    return f

def frame_func_melt( data ):
    f,a = plt.subplots(1,1)
    data.plot.pcolormesh(cmap="RdBu_r")
    return f

def frame_func_twterm( data ):
    f,a = plt.subplots(1,1)
    data.plot.pcolormesh(vmin=-1000,vmax=200,cmap="magma")
    return f
print(ds)
anim = True
if anim:
    xAnimate.make_animation( ds.TWtermav, fp_out = './Twterm.mp4',
                             anim_dim = 'time', fps=2,
                             frame_func = frame_func_twterm)

    xAnimate.make_animation( ds.melt, fp_out = './melt.mp4',
                             anim_dim = 'time',fps=2,
                             frame_func = frame_func_melt)
    ##


    xAnimate.make_animation( ds.Ut, fp_out = './U.mp4',
                            anim_dim = 'time',fps=2,
                            frame_func = frame_func_v)
    ##
    xAnimate.make_animation( ds.U2t, fp_out = './U2.mp4',
                            anim_dim = 'time', fps=2,
                            frame_func = frame_func_v)
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
X,Y = np.meshgrid(ds.x.values/1000,ds.y.values/1000)
U2 = ds.U2t[100:].mean(dim="time").values
V2 = ds.V2t[100:].mean(dim="time").values
plt.pcolormesh(X,Y,ds.melt[100:].mean(dim="time").where(ds.y<150000),vmin=0,vmax=60,cmap="Reds")
cbar = plt.colorbar()
cbar.set_label("(m/yr)",fontsize=16)
box_kernel = Box2DKernel(5,mode="center")

#X = convolve(X, box_kernel,preserve_nan=True)
#Y = convolve(Y, box_kernel,preserve_nan=True)
#
U2 = convolve(U2, box_kernel,preserve_nan=True)
V2 = convolve(V2, box_kernel,preserve_nan=True)

U2[np.logical_or(U2==0,V2==0)] = np.nan
V2[np.logical_or(U2==0,V2==0)] = np.nan
 
plt.quiver(X[::2,::2],Y[::2,::2],U2[::2,::2],V2[::2,::2],scale=1,width=0.0035,color="navy")
plt.xlabel("X (km)",fontsize=16)
plt.ylabel("Y (km)",fontsize=16)
plt.show()
