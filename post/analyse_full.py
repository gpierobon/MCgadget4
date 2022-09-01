import os
import errno
import numpy as np
import MAS_library as MASL
import Pk_library as PKL
import void_library as VL
from snapshot_functions import gadget_to_particles, density_profile, fof_to_halos
from matplotlib.colors import LogNorm
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import cmasher as cmr
import sys


sys.path.insert(0,"/home/561/gp5547/jaxionsdir_v1.1/jaxions/scripts")
from pyaxions import jaxions as pa
import h5py

snap_base = sys.argv[1]
Nfiles    = int(sys.argv[2])
atype     = int(sys.argv[3])
mtype     = sys.argv[4]
vtime     = int(sys.argv[5])
vthres    = float(sys.argv[6])
denfile   = int(sys.argv[7])
nhalos    = int(sys.argv[8])

snapshots = [str(snap_base)+'/snap'+f'_00{i}' for i in range(Nfiles)]  #snapshot base
grid      = 512                     #grid size
ptypes    = [1]                   #CDM + neutrinos
MAS       = 'CIC'                   #Cloud-in-Cell
do_RSD    = False                   #dont do redshif-space distortions
axis      = 0                       #axis along which place RSD; not used here
verbose   = True   #whether print information on the progress
threads   = 4

f = [h5py.File(str(snap)+'.hdf5','r') for snap in snapshots]
L = f[0]['Header'].attrs['BoxSize']
z = [f[i]['Header'].attrs['Redshift'] for i in range(Nfiles)]

BoxSize = L

# Compute the effective number of particles/mass in each voxel
delta = [MASL.density_field_gadget(snap, ptypes, grid, MAS, do_RSD, axis, verbose) for snap in snapshots]
# compute density contrast: delta = rho/<rho> - 1
delta /= np.mean(delta, dtype=np.float64); delta -= 1.0

print('Delta grids are ready')

if atype == 0:
    print("Plotting projectons in aout/plots ...")
    delta += 1
    fig,ax = pa.plot.single()


if atype == 1:
    print("Saving power spectrum data in aout/spectrum ...")
    
    try:
        os.mkdir('aout/spectrum/'+str(snap_base))
    except OSError as exc:
        if exc.errno != errno.EEXIST:
            raise
        pass
   
    Pk  = [PKL.Pk(de, BoxSize, axis, MAS, threads, verbose) for de in delta]
    k   = [Pk[i].k3D for i in range(Nfiles)]
    Pk0 = [Pk[i].Pk[:,0] for i in range(Nfiles)]

    fi = [open('aout/spectrum/'+str(snap_base)+f'/ps_00{i}.txt','w+') for i in range(Nfiles)]
    
    for i in range(Nfiles):
        fi[i].write("#k P(k) Delta^2(k)\n")
        for j in range(len(k[i])):
            fi[i].write("%.5f %.5f %.5f\n"%(k[i][j],Pk0[i][j],Pk0[i][j]*k[i][j]**3/(2*np.pi**2)))
    
    [fi[j].close() for j in range(Nfiles)]

if atype == 2:
    print("Saving delta grids in aout/delta ...")

if atype == 3:
    print("Finding voids as a funcition of radius ...")
    
    try:
        os.mkdir('aout/void/'+str(snap_base))
    except OSError as exc:
        if exc.errno != errno.EEXIST:
            raise
        pass
    
    DeltaGrid  = L/512
    threshold  = -vthres
    Radii      = DeltaGrid*np.array([5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27, 29, 31,
                       33, 35, 37, 39, 41, 43, 45, 47, 49, 51, 53, 55, 57, 59, 61, 63, 65, 67],dtype=np.float32)
    threads1   = 4
    threads2   = 4
    void_field = True

    fv = open('aout/void/'+str(snap_base)+'/thr_'+str(vthres)+'_z_%d.txt'%vtime,'w+') 
   
    V = VL.void_finder(delta[vtime], BoxSize, threshold, Radii, threads1, threads2, void_field=void_field)
    void_pos    = V.void_pos    #positions of the void centers
    void_radius = V.void_radius #radius of the voids
    VSF_R       = V.Rbins       #bins in radius for VSF(void size function)
    VSF         = V.void_vsf    #VSF (#voids/volume/dR)
    if void_field:
        vfield = V.in_void
    vfraction = np.sum(field)/(512**3)

    print("Saving void finding data in aout/void ...")
    for i in range(len(VSF)):
        fv.write('%f %f\n'%(VSF_R[i],VSF[i]))
    fv.close()

if atype == 4:
    print("Finding voids as a function of redshift ...")
    
    try:
        os.mkdir('aout/void/'+str(snap_base))
    except OSError as exc:
        if exc.errno != errno.EEXIST:
            raise
        pass
    
    DeltaGrid  = L/512
    threshold  = -vthres
    #Radii      = DeltaGrid*np.array([9, 11, 13, 15, 17, 19, 21, 23, 25, 27, 29, 31, 33, 35, 
    #                                37, 39, 41, 43, 45, 47, 49, 51, 53, 55, 57, 59, 61, 63, 65, 67],dtype=np.float32)
    Radii      = DeltaGrid*np.array([5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95, 100],dtype=np.float32)
    threads1   = 16
    threads2   = 4
    void_field = True

    fv = open('aout/void/'+str(snap_base)+'/fvoid_thr_'+str(vthres)+'_dens.txt','w+') 
    print("Saving void finding data in aout/void ...")
    
    for i in range(Nfiles):
        V = VL.void_finder(delta[i], BoxSize, threshold, Radii, threads1, threads2, void_field=void_field)
        void_pos    = V.void_pos    #positions of the void centers
        void_radius = V.void_radius #radius of the voids
        VSF_R       = V.Rbins       #bins in radius for VSF(void size function)
        VSF         = V.void_vsf    #VSF (#voids/volume/dR)
        if void_field:
            vfield = V.in_void
            vfraction = np.sum(vfield)/(512**3)
        check       = VL.void_safety_check(np.float32(delta[i]), np.float32(void_pos), np.float32(void_radius), np.float32(BoxSize))
        ov          = check.mean_overdensity 
        fv.write("%f %f %f\n"%(z[i],vfraction,1+np.mean(ov)))
    fv.close()

if atype == 5:
    print("Computing density profiles of MC halos ...")

    try:
        os.mkdir('aout/profile/'+str(snap_base))
    except OSError as exc:
        if exc.errno != errno.EEXIST:
            raise
        pass

    snap = str(snap_base)+'/snap_00'+str(denfile)+'.hdf5'
    fof  = str(snap_base)+'/fof_subhalo_tab_00'+str(denfile)+'.hdf5'
    BoxSize = BoxSize
    pos, _, mass, header = gadget_to_particles(snap)
    hpos, _, hmass, r200,_ = fof_to_halos(fof)
    
    # Sort by mass, not size
    sort = np.argsort(hmass)[::-1]
    hpos = hpos[:,sort]
    hmass = hmass[sort]
    r200 = r200[sort]

    num_halos = nhalos 
    for i in range(num_halos):
        print('Computing profile of group %d ... '%i)
        r,rho = density_profile((pos-hpos[:,i:i+1]),mass,r200[i],BoxSize=BoxSize)
        r = np.array(r)
        rho = np.array(rho)
        fil = open('aout/profile/'+str(snap_base)+'/f_'+str(denfile)+'_gr_'+str(i)+'.txt','w+')
        for j in range(len(r)):
            fil.write('%.5f %.5f\n'%(r[j],rho[j]))
        fil.close()
        
print("done!")




