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

#folderpaths = ['/home/garrett/Projects/laddie/output/ref_2025-02-14_long4x/','/home/garrett/Projects/laddie/output/ref_2025-02-14_long4xcont/']
folderpaths = ['/home/garrett/Projects/laddie/output/ref_2025-02-25/']

ds = dsFromPath(folderpaths)
def vortBudgetBasic(ds):
    f = -1.37e-4
    Cd = 2.5e-3
    t0 = 10
    Ah = 10
    u1 = ds.Uu[t0:].mean(dim="time").values
    v1 = ds.Vv[t0:].mean(dim="time").values
    u2 = ds.U2u[t0:].mean(dim="time").values
    v2 = ds.V2v[t0:].mean(dim="time").values
    dx = (ds.x.values[1]-ds.x.values[0])
    dy = (ds.y.values[1]-ds.y.values[0])
    #h2 = 0.003*(ds.D2[t0:].mean(dim="time").values+bed)+ds.RL[t0:].mean(dim="time").values# + ds.rl[75:].mean(dim="time").values
    h2 = ds.D2[t0:].mean(dim="time").values
    h = ds.D[t0:].mean(dim="time").values
    #M = 0.03*(ds.D2[t0:].mean(dim="time").values+bed)
    fig, (ax1,ax2,ax3) = plt.subplots(1,3)
    dhdx = (-h2+np.roll(h2,-1,axis=1))/dx
    dhdy = (-h2+np.roll(h2,-1,axis=0))/dy
    u2mag = np.sqrt(u2**2+v2**2)
    curl = (-u2*u2mag+np.roll(u2*u2mag,-1,axis=0))/dy
    curl2 = (-v2*u2mag+np.roll(v2*u2mag,-1,axis=1))/dx
    curl = -curl+curl2
    X,Y = np.meshgrid(ds.x.values,ds.y.values)

    fig, ((ax1,ax2,ax5),(ax3,ax4,ax6)) = plt.subplots(2,3)
    ax1.pcolormesh(X,Y,f*(ds.ent2+ds.entr-ds.detr)[t0:].mean(dim="time")/(60*60*24*365),vmin=-8e-9,vmax=8e-9,cmap="RdBu_r")
    #ax2.pcolormesh(X,Y,intface,vmin=-8e-7,vmax=8e-7,cmap="RdBu_r")
    ax2.pcolormesh(X,Y,(u2*dhdx + v2*dhdy)*(f**1),vmin=-8e-8,vmax=8e-8,cmap="RdBu_r")
    ax3.pcolormesh(X,Y,Cd*curl,vmin=-8e-9,vmax=8e-9,cmap="RdBu_r")

    dudx = (-u2+np.roll(u2,-1,axis=1))/dx
    dvdy = (-v2+np.roll(v2,-1,axis=0))/dy
    hudx = h2*dudx
    hvdy = h2*dvdy

    dyhvdy = (-hvdy+np.roll(hvdy,-1,axis=0))/dy
    dxhudx = (-hudx+np.roll(hudx,-1,axis=0))/dx

    dyhvdy = (dyhvdy+np.roll(dyhvdy,1,axis=1))/dy
    dxhudx = (dxhudx+np.roll(dxhudx,1,axis=0))/dx

    ax4.pcolormesh(X,Y,Ah*(dyhvdy+dxhudx),vmin=-8e-9,vmax=8e-9,cmap="RdBu_r")

    H = h+h2

    Uvvisc = (u1-u2)*2/((H+np.roll(H,-1,axis=1))/2)
    Vvvisc = (v1-v2)*2/((H+np.roll(H,-1,axis=0))/2)

    Avcurl = (1e-4)*(-(-Uvvisc-np.roll(Uvvisc,-1,axis=0))/dy + (-Vvvisc-np.roll(Vvvisc,-1,axis=1))/dx)
    ax5.pcolormesh(X,Y,Avcurl,vmin=-8e-9,vmax=8e-9,cmap="RdBu_r")

    plt.show()

def vortBudget(ds):
    t0=7
    fig, axises = plt.subplots(3,4)
    titles = ['DUdt','conv(U)', 'D grad(M)','fD<V,U>','Cd U|U|','Ah lap(V)','Av(U-U2)/H','D grad(M) + D grad(pi)','w V','sponge','D grad(pi)']
    for i in range(11):     
        ds.vortterms[t0:].mean(dim="time")[i].plot.pcolormesh(ax=axises.flatten()[i],vmin=-5e-8,vmax=5e-8,cmap="RdBu_r")
        axises.flatten()[i].set_title(titles[i])
    dx = (ds.x.values[1]-ds.x.values[0])
    dy = (ds.y.values[1]-ds.y.values[0])
    X,Y = np.meshgrid(ds.x.values,ds.y.values)
    
    c = axises.flatten()[-1].pcolormesh(X,Y,ds.vortterms.sum(axis=1).mean(dim="time"),vmin=-5e-8,vmax=5e-8,cmap="RdBu_r")
    c = axises.flatten()[7].pcolormesh(X,Y,(ds.vortterms[:,2]+ds.vortterms[:,10]).mean(dim="time"),vmin=-5e-8,vmax=5e-8,cmap="RdBu_r")
    axises.flatten()[-1].set_title("Vorticity sum")
    axises.flatten()[7].set_title(titles[7])
    plt.colorbar(c,ax=axises.flatten()[-1])
    plt.show()

def simpleVortBudget(ds):
    t0=2
    fig, ((ax1,ax2),(ax3,ax4)) = plt.subplots(2,2)
    titles = ['conv(U)', 'fD<V,U>','Cd U|U|','D grad(pi) + D grad(M)']

    ds.vortterms[t0:].mean(dim="time")[1].plot.pcolormesh(ax=ax1,vmin=-5e-9,vmax=5e-9,cmap="RdBu_r")
    ax1.set_title(titles[0])

    ds.vortterms[t0:].mean(dim="time")[3].plot.pcolormesh(ax=ax2,vmin=-5e-9,vmax=5e-9,cmap="RdBu_r")
    ax2.set_title(titles[1])

    ds.vortterms[t0:].mean(dim="time")[4].plot.pcolormesh(ax=ax3,vmin=-5e-9,vmax=5e-9,cmap="RdBu_r")
    ax3.set_title(titles[2])

    (ds.vortterms[:,2]+ds.vortterms[:,10])[t0:].mean(dim="time").plot.pcolormesh(ax=ax4,vmin=-5e-9,vmax=5e-9,cmap="RdBu_r")
    ax4.set_title(titles[3])

    plt.show()

def PVplot(ds):
    f = -1.37e-4
    Cd = 2.5e-3
    t0=10
    u2 = ds.U2u[t0:].mean(dim="time").values
    v2 = ds.V2v[t0:].mean(dim="time").values
    dx = (ds.x.values[1]-ds.x.values[0])
    dy = (ds.y.values[1]-ds.y.values[0])
    h2 = ds.D2[t0:].mean(dim="time").values
    X,Y = np.meshgrid(ds.x.values,ds.y.values)
    relvort = (np.roll(v2,-1,axis=1)-np.roll(v2,1,axis=1))/dx - (np.roll(u2,-1,axis=0)-np.roll(u2,1,axis=0))/dy
    fig, (ax1,ax2) = plt.subplots(1,2)
    ax1.pcolormesh(X,Y,relvort,vmin=-8e-6,vmax=8e-6,cmap="RdBu_r")
    ax2.pcolormesh(X,Y,(f+relvort)/h2,vmin=-4e-6,vmax=4e-6,cmap="RdBu_r")
    plt.show()


 
#PVplot(ds)

#simpleVortBudget(ds)
vortBudget(ds)


#dsbed = xr.open_dataset("../input/mitgcm.nc")
#M = ds.RL[t0:].mean(dim="time").values*9.8+ds.TWtermav[t0:].mean(dim="time")#0.0003*(ds.D2[t0:].mean(dim="time").values+bed)
# M = ds.RL[t0:].mean(dim="time").values+ds.TWtermav[t0:].mean(dim="time")#0.0003*(ds.D2[t0:].mean(dim="time").values+bed)
# M = 0*ds.RL[t0:].mean(dim="time").values+0*ds.TWtermav[t0:].mean(dim="time")#0.0003*(ds.D2[t0:].mean(dim="time").values+bed)
# M = (ds.RL[t0:].mean(dim="time").values + ds.TWtermav[t0:].mean(dim="time").values)



# dMdx = (-M+np.roll(M,-1,axis=1))/dx
# dMdy = (-M+np.roll(M,-1,axis=0))/dy

# honu = (h2+np.roll(h2,-1,axis=1))/2

# honv = (h2+np.roll(h2,-1,axis=0))/2

# intface = -((-dMdx*honu) + np.roll((dMdx*honu),-1,axis=0))/dy + ((-dMdy*honv) + np.roll((dMdy*honv),-1,axis=1))/dx

