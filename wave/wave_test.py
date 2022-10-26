import numpy as np
import matplotlib.pyplot as plt
import plot_utils

gather =np.load('wave_data/gather_00000000_00000000.npy')
gather_raw =np.load('wave_data/gather_raw_00000000_00000000.npy')
velocity =np.load('wave_data/velocity_00000000_const.npy')
wavefields = np.load('wave_data/wavefields_00000000_00000000.npy')
wavefields2 = np.load('wave_data/wavefields_00000000_00000000.npy')
#wavefields = np.load('wave_data/wavefields_00000000_00000000_const.npy')

print(velocity.shape)
print(gather.shape)
print(gather_raw.shape)
print(wavefields.shape)



VEL_RUN = "layers_validate"
SIM_RUN = "layers_2ms_validate"
DELTARECi = 10
isim=(600,0)
#VEL_DIR =  ""#"velocity/" + VEL_RUN + "/"
#OUT_SIM_DIR = ""#"gather/" + SIM_RUN + "/"
#wavefields = np.load(OUT_SIM_DIR + "wavefields_%.8i_%.8i.npy"%(isim[0],isim[1]))
wavefields = wavefields[::4]
wavefields2 = wavefields2[::4]
#gather = np.load(OUT_SIM_DIR + "gather_%.8i_%.8i.npy"%(isim[0],isim[1]))
#velocity = np.load(VEL_DIR + "velocity_%.8i.npy"%(isim[0]))
#reflectivity = np.load(VEL_DIR + "reflectivity_%.8i.npy"%(isim[0]))
source_is = np.load("wave_data/source_is.npy")
receiver_is = np.load("wave_data/receiver_is.npy")
print(source_is)
print(source_is.shape)
source_i = source_is[0,0,:]
NREC = 1#len(receiver_is)



DT = 0.002
T_GAIN = 2.5
DX,DZ = 5., 5.
NX,NZ = 300, 300#128, 128
NSTEPS = 512

VLIM = 0.6
CLIM = (1500,3600)

gain = np.arange(NSTEPS)**T_GAIN
gain = gain / np.median(gain)

print(np.arange(0,NZ,40)[::-1])
print((DZ*np.arange(0,NZ,40)[::-1]).astype(np.int))
plt.figure(figsize=(11,6))

# velocity & wavefield
plt.subplot2grid((2, 2), (0, 0), colspan=1)
plt.imshow(velocity.T, alpha=0.6, cmap="gray_r", vmin=CLIM[0], vmax=CLIM[1])
plt.imshow(wavefields[50].T, aspect=1, cmap=plot_utils.rgb, alpha=0.7, vmin = -2, vmax=2)
cb = plt.colorbar()
cb.ax.set_ylabel('Pressure (arb)')
plt.scatter(source_i[0], source_i[1], color='black', s=120)
plt.scatter(receiver_is[:,0], receiver_is[:,1], color='red', marker='v', s=60)
plt.gca().set_anchor('C')# centre plot
plt.yticks(np.arange(0,NZ,40)[::-1], (DZ*np.arange(0,NZ,40)[::-1]).astype(np.int))
plt.xticks(np.arange(0,NX,40), (DX*np.arange(0,NX,40)).astype(np.int))
plt.ylabel("Depth (m)")
plt.xlim(0, NX-1)
plt.ylim(NZ-1, 0)


# velocity
plt.subplot2grid((2, 2), (1, 0), colspan=1)
plt.imshow(velocity.T, vmin=CLIM[0], vmax=CLIM[1])
cb = plt.colorbar()
cb.ax.set_ylabel('Velocity ($\mathrm{ms}^{-1}$)')
plt.scatter(receiver_is[:,0], receiver_is[:,1], color='red', marker='v', s=60)
plt.gca().set_anchor('C')# centre plot
plt.yticks(np.arange(0,NZ,40)[::-1], (DZ*np.arange(0,NZ,40)[::-1]).astype(np.int))
plt.xticks(np.arange(0,NX,40), (DX*np.arange(0,NX,40)).astype(np.int))
plt.xlabel("Distance (m)")
plt.ylabel("Depth (m)")
plt.xlim(0, NX-1)
plt.ylim(NZ-1, 0)

# gather
lim = 1.
plt.subplot2grid((2, 2), (0, 1), rowspan=2)
for ir in range(NREC): plt.plot(lim*ir+gain*gather[ir,:], DT*np.arange(NSTEPS), color='tab:red')
plt.gca().invert_yaxis()
plt.xlabel("Receiver offset (m)")
plt.xticks(np.arange(0,NREC*lim,lim), [int(-(NREC-1)*DELTARECi*DX/2 + irec*DELTARECi*DX) for irec in range(NREC)])
plt.ylabel("Two way time (s)")
plt.ylim(DT*400,0)
plt.xlim(-lim, lim*(NREC-1)+lim+0.3)
plt.gca().set_anchor('C')# centre plot
plt.xticks(rotation=30)

plt.subplots_adjust(left=0.2, right=0.9, bottom=0.1, top=1., hspace=0.1, wspace=0.3)

plt.savefig("fig03.pdf", bbox_inches='tight', pad_inches=0.01, dpi=300)




