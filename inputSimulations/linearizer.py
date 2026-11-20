import numpy as np
import matplotlib.pyplot as plt
from matplotlib import image
import os

def moving_average(vec,n):
    mask=np.ones(n)/n
    smooth=np.convolve(vec,mask,mode="same")
    smooth[:n//2]=smooth[n//2]
    smooth[-n // 2:] = smooth[-n // 2]
    return smooth


small=0

pix=1 / 0.96
S_fac=1e6*0.2517


ill_map=image.imread("inputSimulations//illumination_lin_180µW.tiff").astype(float)
map1=np.sum(ill_map)
map2=np.sum(image.imread("inputSimulations//fluoTest2.tiff").astype(float))


Lx=len(ill_map[0,:])
Ly=len(ill_map[:,0])

C=[Lx/2,Ly/2]
Ldia=int(np.abs(C[0]))-1

xgrid, ygrid=np.meshgrid(range(Lx),range(Ly))

dists=np.abs(xgrid-C[0])#np.abs(ygrid-C[1])

ill_vec=np.zeros(Ldia)

for i in range(Ldia):
    ill_vec[i]=np.mean(ill_map[(dists>=i) & (dists<i+1)])


inputname = "inputSimulations//radialEnergyDensityMap_180µW"
outputname = "inputSimulations//linearEnergyDensityMap_180µW_good"
Soutputname = "inputSimulations//linearEnergyDensityMap_Smodel_180µW_good"
ratio = map2 / map1

radial_int = np.genfromtxt(inputname+".txt",skip_header=1)
ill_vec = ill_vec / np.max(ill_vec) * np.max(radial_int)*ratio
smooth_ill = moving_average(ill_vec,40)
s_smooth = smooth_ill*S_fac

if os.path.isfile(outputname+".txt"):
    os.remove(outputname+".txt")
with open(outputname+".txt","a") as output:
    output.write("intensityVsPixelDistance\n")
    for val in smooth_ill:
        output.write(f"{val}\n")

if os.path.isfile(Soutputname+".txt"):
    os.remove(Soutputname+".txt")
with open(Soutputname+".txt","a") as output:
    output.write("intensityVsPixelDistance\n")
    for val in s_smooth:
        output.write(f"{val}\n")

plt.plot(ill_vec)
#plt.plot(radial_int)
plt.plot(smooth_ill)
plt.yscale("log")
plt.show()