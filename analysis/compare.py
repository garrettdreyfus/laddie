import xarray as xr
import matplotlib.pyplot as plt
import glob
import re
import math
from pathlib import Path
import xAnimate
import cmocean.cm as cm


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

folderpaths = ['/home/garrett/Projects/laddie-original/output/NoDetr_2025-01-14_0/']
original = dsFromPath(folderpaths)

folderpaths = ['/home/garrett/Projects/laddie/output/NoDetr_2025-01-14/']
twolayer = dsFromPath(folderpaths)

#fig, ((ax1,ax2),(ax3,ax4),(ax5,ax6)) = plt.subplots(3,2)
#
#original.Ut[3:].mean(dim="time").plot.pcolormesh(ax=ax1,vmin=-0.75,vmax=0.75,cmap="RdBu_r")
#original.Vt[3:].mean(dim="time").plot.pcolormesh(ax=ax3,vmin=-0.75,vmax=0.75,cmap="RdBu_r")
#original.melt[3:].mean(dim="time").plot.pcolormesh(ax=ax5,vmin=0,vmax=110)
#
#twolayer.Ut[3:].mean(dim="time").plot.pcolormesh(ax=ax2,vmin=-0.75,vmax=0.75,cmap="RdBu_r")
#twolayer.Vt[3:].mean(dim="time").plot.pcolormesh(ax=ax4,vmin=-0.75,vmax=0.75,cmap="RdBu_r")
#twolayer.melt[3:].mean(dim="time").plot.pcolormesh(ax=ax6,vmin=0,vmax=110)

#plt.show()
#
#fig, ((ax1,ax2),(ax3,ax4),(ax5,ax6)) = plt.subplots(3,2)
#original.T[50:].mean(dim="time").plot.pcolormesh(ax=ax1,cmap = cm.thermal)
#original.S[50:].mean(dim="time").plot.pcolormesh(ax=ax3,cmap = cm.haline)
##original.D[50:].mean(dim="time").plot.pcolormesh(ax=ax5,vmin=10,vmax=120,cmap = cm.deep)
#
#twolayer.T[50:].mean(dim="time").plot.pcolormesh(ax=ax2,cmap = cm.thermal)
#twolayer.S[50:].mean(dim="time").plot.pcolormesh(ax=ax4,cmap = cm.haline)
#twolayer.D[50:].mean(dim="time").plot.pcolormesh(ax=ax6,vmin=10,vmax=120,cmap = cm.deep)
#plt.show()


original.D.mean(dim=['x','y']).plot(label="original")
twolayer.D.mean(dim=['x','y']).plot(label="++")
plt.legend()
plt.show()


