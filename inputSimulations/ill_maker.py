import numpy as np
import matplotlib.pyplot as plt
from numpy.random import default_rng
from scipy.signal import fftconvolve
from scipy.optimize import curve_fit
from matplotlib import image 
from scipy.interpolate import interp1d
from scipy.signal import convolve2d

def gauss(x, s, m, A):
    return A*1/(s*np.sqrt(2*np.pi))*np.exp(-(x-m)**2/(2*s**2))


def neg_exp(x,b):
    return 1-np.exp(+x*b)

intensity=610                                               #intensity from experimental data (ring)
newint=610                                                  #make simulation map with different intensity wrt experiments
gaussian=False                                              #intensity from experimental data (gaussian)
symmetrized=False                                           #not used 
Smodel=False                                                #create the illumination map based for S-model
factor = 1e6 * 0.2517                                       #conversion factor from normal to S-model
linear=False                                                #intensity from experimental data (linear) 
large=True                                                  #intensity from experimental data (large ring)
pix=1.041667                                                #pixel in micrometers
inputname="inputSimulations//radialEnergyDensityMap"        #input folder of experimental data
outputnamei="tbr_fields//ill_map"                           #output folder of illumination map
outputnamet="tbr_fields//tbr_map"                           #output folder of tumbling rate map
samepeak=False                                              #instead of normalizing the 
normalisationPower=0                                        #not used

suffix1=""
suffix2=""


if large:
    suffix1 += "_large_ring"
    suffix2 += "_large_ring"
elif gaussian:
    suffix1+="_GAU"
    suffix2+="_gauss"
elif symmetrized:
    suffix2+="_symm"
elif linear:
    print("linear!")
    #suffix1 += "_lin"
    #suffix2 += "_lin"
    #normalisationPower=140
else:
    suffix2 += "_ring"
if Smodel:
    suffix1 += "_Smodel"
    suffix2 += "_Smodel"

if samepeak and large:
   suffix2+=f"_samepeak" 
    
suffix1+=f"_{intensity}µW_good"
suffix2+=f"_{newint}muW"



if not linear:
    radial_int = np.genfromtxt(inputname+suffix1+".txt",skip_header=1)
    radial_int*=newint/intensity
    if samepeak and large:
        norm=0.0010939796977177726/0.0004875693192653481
        radial_int*=norm
    if symmetrized:
        top = np.argmax(radial_int)
        symm_radial_int = np.append(radial_int[top:2 * top][::-1], radial_int[top:])
        radial_int = np.copy(symm_radial_int)
    int_converter = interp1d(np.arange(len(radial_int))*pix, radial_int, kind='cubic',fill_value=0,bounds_error=False)
    pix_L = int(len(radial_int)/np.sqrt(2))-1
    if large:
        pix_L = int(1.5*pix_L)

    #plt.plot(np.arange(len(radial_int))*pix, radial_int)
    #plt.plot(np.arange(len(radial_int))*pix, int_converter(np.arange(len(radial_int))*pix))
    #plt.plot(np.arange(len(radial_int))*pix,gauss(np.arange(len(radial_int)),35,193,.098)+gauss(np.arange(len(radial_int)),35,-193,.099))
    #plt.yscale("log")
    #plt.show()

    ix = np.arange(-pix_L, pix_L+1)*pix
    jy = np.arange(-pix_L, pix_L+1)*pix

    # Generate the meshgrid
    ixv, jyv = np.meshgrid(ix, jy)
    ij_dist=np.sqrt(ixv**2+jyv**2)
    ill_map=int_converter(ij_dist)

else:
    #ill_map = image.imread("inputSimulations//illumination"+suffix1+".tiff").astype(float).transpose()
    #ill_map = ill_map* normalisationPower / ill_map.sum()
    #print(ill_map.max())
    inputname="inputSimulations//linearEnergyDensityMap"
    linear_int = np.genfromtxt(inputname+suffix1+".txt",skip_header=1)
    linear_int*=newint/intensity
    int_converter = interp1d(np.arange(len(linear_int))*pix, linear_int, kind='cubic',fill_value=0,bounds_error=False)
    pix_L = int(len(linear_int))-1

    #plt.plot(np.arange(len(linear_int))*pix, linear_int)
    #plt.plot(np.arange(len(linear_int))*pix, int_converter(np.arange(len(linear_int))*pix))
    #plt.plot(np.arange(len(radial_int))*pix,gauss(np.arange(len(radial_int)),35,193,.098)+gauss(np.arange(len(radial_int)),35,-193,.099))
    #plt.yscale("log")
    #plt.show()

    ix = np.arange(-pix_L, pix_L+1)*pix
    jy = np.arange(-pix_L, pix_L+1)*pix

    # Generate the meshgrid
    ixv, jyv = np.meshgrid(ix, jy)
    ij_dist=np.abs(ixv)
    ill_map=int_converter(ij_dist)
    radial_int=linear_int

tumble_data = np.genfromtxt("inputSimulations//tumblingVsEnergyDensity.txt",skip_header=1)
tumble_converter = interp1d(tumble_data[:,2], tumble_data[:,1], kind='slinear',fill_value="extrapolate")


#tbs=np.linspace(0,0.002,num=1000)
#vtb=425#511
#plt.plot(tbs,tumble_converter(tbs),color="b")
#plt.plot(tumble_data[:,2],tumble_data[:,1],color="r")
#plt.show()

#plt.plot(np.arange(len(radial_int))*pix, tumble_converter(radial_int))
#plt.plot(np.arange(len(radial_int))*pix, radial_int)
#plt.axvline(vtb)
#plt.axhline(tumble_converter(radial_int)[int(511/pix)])
#print(tumble_converter(radial_int)[int(511/pix)])
tval=3.0889#2.679346076176182#3.0708352

threshold_dist=np.argmin(np.abs(tumble_converter(radial_int)-tval))*pix
max_int=np.argmax(np.abs(tumble_converter(radial_int)))*pix
travel_dist=np.abs(threshold_dist-max_int)
print(f"threshold: {threshold_dist} mum")
print(f"max_int: {max_int} mum")
print(f"travel_dist: {travel_dist} mum")
print(f"max_int_val: {np.max(radial_int)} I")
#plt.axvline(threshold_dist,c="r")
#plt.axhline(3.0708352,c="r")
#plt.axvline(max_int,c="g")
#plt.axhline(radial_int[np.argmin(np.abs(tumble_converter(radial_int)-tval))],c="r")
#plt.show()
#plt.plot(np.arange(len(radial_int))*pix, tumble_converter(radial_int))
#rat=tumble_data[-1,1]-tumble_data[0,1]
#off=tumble_data[0,1]
#plt.plot(tumble_data[:,2],(tumble_data[:,1]-off)/rat,color="r")
#param,pcov=curve_fit(neg_exp,tumble_data[:,2],(tumble_data[:,1]-off)/rat,p0=[-5])#,p0=[ 3.06973321e+00,1.13406752e+00,-1.30862741e+05])
#print(param)
#perr=np.sqrt(np.diag(pcov))
#plt.plot(tbs,neg_exp(tbs,-50000),linestyle='--',c="b")
#plt.xscale("log")

#tumble_map=neg_exp(ill_map,*param)
if Smodel:
    tumble_map=tumble_converter(ill_map/factor)
else:
    tumble_map=tumble_converter(ill_map)

print(f"tbr min: {np.min(tumble_map)}")


Nx=len(ill_map[:,0])
Ny=len(ill_map[0,:])
avg_int=np.mean(ill_map)
avg_tumb=np.mean(tumble_map)
print(Nx)
print(Ny)
nplots=2
fig,ax=plt.subplots(nrows=1,ncols=nplots,figsize=(nplots*10,8))

ax[0].pcolormesh(ill_map.transpose(),cmap="afmhot",vmin=np.min(ill_map),vmax=np.max(ill_map))
ax[1].pcolormesh(tumble_map.transpose(),cmap="plasma",vmin=np.min(tumble_map),vmax=np.max(tumble_map))
ax[0].set_title("illumination")
ax[1].set_title("tumbling rate")
plt.show()

with open(outputnamei+suffix2+".dat","w") as txtfile:
    txtfile.write(f"{Nx} {Ny} {avg_int} {avg_tumb} {pix}\n")

with open(outputnamei+suffix2+".dat","a") as txtfile:    
    np.savetxt(txtfile,ill_map)
    
with open(outputnamet+suffix2+".dat","w") as txtfile:
    txtfile.write(f"{Nx} {Ny} {avg_int} {avg_tumb} {pix}\n")

with open(outputnamet+suffix2+".dat","a") as txtfile:    
    np.savetxt(txtfile,tumble_map)
