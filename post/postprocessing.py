import os,sys,errno
import numpy as np
import h5py as h5

import MAS_library as MASL
import Pk_library as PKL
import void_library as VL

import yt
import matplotlib.pyplot as plt
import cmasher as cmr
from matplotlib import cm
from matplotlib.colors import LogNorm

sys.path.insert(0,"/home/561/gp5547/jaxionsdir_v1.1/jaxions/scripts")
sys.path.insert(0,"/home/561/gp5547/MCgadget4/post")

#from post import gadget as ga
import gadget as ga 
from pyaxions import jaxions as pa

snap_base = sys.argv[1]
atype     = int(sys.argv[2])
Nfiles    = int(sys.argv[3])
stime     = int(sys.argv[4])
mtype     = sys.argv[5]
camres    = int(sys.argv[6])
nbins     = int(sys.argv[7])
dflag     = int(sys.argv[8])
rlist     = int(sys.argv[9])
vthres    = float(sys.argv[10])
fof_flag  = sys.argv[11]
rad_type  = sys.argv[12]
r_min     = float(sys.argv[13])
nhalos    = int(sys.argv[14])

if Nfiles > 1:
    snapshots = [str(snap_base)+'/snap'+f'_00{i}' for i in range(Nfiles)]
    f = [h5.File(str(snap)+'.hdf5','r') for snap in snapshots]
    z = [f[i]['Header'].attrs['Redshift'] for i in range(Nfiles)]
    L = f[0]['Header'].attrs['BoxSize']
else:
    snapshot  = str(snap_base)+'/snap'+f'_00{stime}'
    f = h5.File(str(snapshot)+'.hdf5','r')
    z = f['Header'].attrs['Redshift']
    L = f['Header'].attrs['BoxSize']

BoxSize   = L
grid      = 512                 
ptypes    = [1]                   
MAS       = 'CIC'               
verbose   = True
threads   = 4

# CIC creation of delta via MASL library
if Nfiles > 1:
    delta = [MASL.density_field_gadget(snap, ptypes, grid, MAS, do_RSD=False, axis=0, verbose=verbose) for snap in snapshots]
else:
    delta = MASL.density_field_gadget(snapshot, ptypes, grid, MAS, do_RSD=False, axis=0, verbose=verbose) 
    
delta /= np.mean(delta, dtype=np.float64) 
delta -= 1.0

##########################################################################
if atype == 0:

    print("Plotting projectons in aout/plots ...")
    ga.check_folder('plots',snap_base)
    delta += 1

    if Nfiles > 1:
        for i in range(Nfiles):
            fig,ax = pa.plot.single(size_x=13,size_y=12)
            im = ax.imshow(np.log10(np.mean(delta[i],axis=0)),cmap='cmr.%s'%mtype,vmax=2,vmin=-2,origin='lower',extent=[-L/2,L/2,-L/2,L/2]); pa.plot.cbar(im)
            ax.set_xlabel(r'$x~({\rm pc}/h)$'); ax.set_ylabel(r'$y~({\rm pc}/h)$')
            fig.savefig('aout/plots/'+str(snap_base)+'/z%.1f_%s.pdf'%(z[i],mtype),bbox_inches='tight')
    else:
        fig,ax = pa.plot.single(size_x=13,size_y=12)
        im = ax.imshow(np.log10(np.mean(delta,axis=0)),cmap='cmr.%s'%mtype,vmax=2,vmin=-2,origin='lower',extent=[-L/2,L/2,-L/2,L/2]); pa.plot.cbar(im)
        ax.set_xlabel(r'$x~({\rm pc}/h)$'); ax.set_ylabel(r'$y~({\rm pc}/h)$')
        fig.savefig('aout/plots/'+str(snap_base)+'/z%.1f_%s.pdf'%(z,mtype),bbox_inches='tight')
        
##########################################################################
if atype == 1:
    
    print('Saving volume rendering in aout/render ...')
    ga.check_folder('render',snap_base)

    res  = camres
    bmin = 1e-6
    bmax = 1e3
    
    if Nfiles > 1:
        for i in range(Nfiles):
            ga.vol_render(delta[i],res,b,bmax,snap_base,i)
    else:
        ga.vol_render(delta,res,b,bmax,snap_base,stime)

##########################################################################
if atype == 2:

    print("Saving power spectrum data in aout/spectrum ...")
    ga.check_folder('spectrum',snap_base)
    
    if Nfiles > 1:
        Pk  = [PKL.Pk(de, BoxSize, axis, MAS, threads, verbose) for de in delta]
        k   = [Pk[i].k3D for i in range(Nfiles)]
        Pk0 = [Pk[i].Pk[:,0] for i in range(Nfiles)]
        fi = [open('aout/spectrum/'+str(snap_base)+f'/ps_00{i}.txt','w+') for i in range(Nfiles)]
        for i in range(Nfiles):
            fi[i].write("#k P(k) Delta^2(k)\n")
            for j in range(len(k[i])):
                fi[i].write("%.5f %.5f %.5f\n"%(k[i][j],Pk0[i][j],Pk0[i][j]*k[i][j]**3/(2*np.pi**2)))
    
        [fi[j].close() for j in range(Nfiles)]
    else:
        raise Exception('For the spectra it is convenient to select multiple files')

##########################################################################
if atype == 3:
    print("Saving delta grids in aout/delta ...")
    ga.check_folder('delta',snap_base)
    
    if Nfiles > 1:
        for i in range(Nfiles):
            f = h5.File('aout/delta/'+str(snap_base)+f'/delta_{i}.hdf5', "w")
            dset = f.create_dataset("density",data=delta[i])
            f.close()
    else:
        f = h5.File('aout/delta/'+str(snap_base)+f'/delta_{stime}.hdf5', "w")
        dset = f.create_dataset("density",data=delta)
        f.close()

##########################################################################
if atype == 4:
    print("Saving density distribution data")
    ga.check_folder('dist',snap_base)
    
    if Nfiles == 1:
        raise Exception('For the delta distribution it is convenient to select multiple files')

    delta += 1
    inc = 1
   
    for i in range(Nfiles):
        h,db = np.histogram(np.log10(delta[i].flatten()),nbins,range=[-6,6])
        dc = (db[1:]+db[0:-1])/2
        
        if dflag:
            dx = np.roll(delta[i],inc,axis=0)-delta[i]
            dy = np.roll(delta[i],inc,axis=1)-delta[i]
            dz = np.roll(delta[i],inc,axis=2)-delta[i]
            diff = np.sqrt(dx**2+dy**2+dz**2) 
            h2,db2 = np.histogram(np.log10(diff.flatten()),nbins,range=[-6,6])
            dc2 = (db[1:]+db[0:-1])/2

        fvi = open('aout/dist/'+str(snap_base)+'/dist_'+str(i)+'.txt','w+') 
        for j in range(len(dc)):
            if dflag:
                fvi.write('%.5f %.5f %.5f %.5f\n'%(10**dc[j],h[j],10**dc2[j],h2[j]))
            else:
                fvi.write('%.5f %.5f\n'%(10**dc[j],h[j]))
        fvi.close()

##########################################################################
if atype == 5:
    print('Saving density distribution data for void centers')
    ga.check_folder('dist',snap_base)
    
    if Nfiles > 1:
        raise Exception('Void density distribution is only available for Nfiles=1')
    
    delta += 1
    inc = 1 
    
    DeltaGrid  = L/512
    threshold  = -vthres
    rad_list   = [rlist]
    Radii      = DeltaGrid*np.array(rad_list, dtype=np.float32)
    threads1   = 16
    threads2   = 4
    
    V = VL.void_finder(delta, BoxSize, threshold, Radii, threads1, threads2, void_field=False)
    void_pos    = V.void_pos
    
    cmx = [int(np.round(void_pos[i,0]/L*512)) for i in range(len(void_pos))]
    cmy = [int(np.round(void_pos[i,1]/L*512)) for i in range(len(void_pos))]
    cmz = [int(np.round(void_pos[i,2]/L*512)) for i in range(len(void_pos))]

    dist = [delta[cmx[i],cmy[i],cmz[i]] for i in range(len(void_pos))]
    dist = np.array(dist)
    
    np.savetxt('aout/void/'+str(snap_base)+'/vdist_r%d_z%d.txt'%(rad_list[0],stime),dist)
    
    if dflag:
        raise Exception('Diff for voids not implemented yet')
        #dx = np.roll(delta[i],inc,axis=0)-delta[i]
        #dy = np.roll(delta[i],inc,axis=1)-delta[i]
        #dz = np.roll(delta[i],inc,axis=2)-delta[i]
        #diff =     



##########################################################################
if atype == 6:
    print("Compute void size function ...")
    ga.check_folder('void',snap_base)
    
    if Nfiles > 1:
        raise Exception('Void size function is only available for Nfiles=1')
    
    DeltaGrid  = L/512
    threshold  = -vthres
    Radii      = DeltaGrid*np.arange(5,67+1,2, dtype=np.float32)
    threads1   = 16
    threads2   = 4
    void_field = True

    fv  = open('aout/void/'+str(snap_base)+'/vsf_thr_'+str(vthres)+'_z%d.txt'%stime,'w+') 
   
    V = VL.void_finder(delta[stime], BoxSize, threshold, Radii, threads1, threads2, void_field=void_field)
    void_pos    = V.void_pos    
    void_radius = V.void_radius 
    VSF_R       = V.Rbins   
    VSF         = V.void_vsf

    print("Saving void finding data in aout/void ...")
    for i in range(len(VSF)):
        fv.write('%f %f\n'%(VSF_R[i],VSF[i]))
    fv.close()


##########################################################################
if atype == 7:
    print("Finding voids volume and density as function of redshift ...")
    ga.check_folder('void',snap_base)
    
    if Nfiles == 1:
        raise Exception('Nfiles>1 is needed!')
    
    DeltaGrid  = L/512
    threshold  = -vthres
    Radii      = DeltaGrid*np.arange(5,100+1,5, dtype=np.float32)
    threads1   = 16
    threads2   = 4
    void_field = True

    fv = open('aout/void/'+str(snap_base)+'/fvoid_thr_'+str(vthres)+'_dens.txt','w+') 
    print("Saving void finding data in aout/void ...")
    
    for i in range(Nfiles):
        V = VL.void_finder(delta[i], BoxSize, threshold, Radii, threads1, threads2, void_field=void_field)
        void_pos    = V.void_pos    
        void_radius = V.void_radius 
        VSF_R       = V.Rbins       
        VSF         = V.void_vsf
        if void_field:
            vfield = V.in_void
            vfraction = np.sum(vfield)/(512**3)
        check       = VL.void_safety_check(np.float32(delta[i]), np.float32(void_pos), np.float32(void_radius), np.float32(BoxSize))
        ov          = check.mean_overdensity 
        fv.write("%f %f %f\n"%(z[i],vfraction,1+np.mean(ov)))
    fv.close()


##########################################################################
if atype == 8:
    print("Computing density profiles of MC halos ...")
    ga.check_folder('profile',snap_base)
    
    if Nfiles > 1:
        raise Exception('Density profile estimation is only available for one file, set Nfiles=1')

    snap = str(snap_base)+'/snap_00'+str(stime)
    f_fof  = str(snap_base)+'/fof_subhalo_tab_00'+str(stime)
    
    with h5.File(f_fof+'.hdf5','r') as fof_file:
        soft_length = fof_file['Parameters'].attrs['SofteningComovingClass1']  
    additional  = False

    head, pp    = ga.load_particles(snap,verbose=False) # pp holds pos,vel,mass,ID
    hfof, halos = ga.load_halos(f_fof,fof=fof_flag,radius=rad_type,additional=additional,verbose=True) # halos holds pos,vel,mass,rad,size(+veldisp,spin)
    
    pos       = pp[0] 
    mass      = pp[2]
    halopos   = halos[0]
    halomass  = halos[2]
    rad       = halos[3]
    num_halos = nhalos 
    
    mask      = np.argsort(halomass)[::-1]
    halopos   = halopos[mask,:]
    halomass  = halomass[mask]
    rad       = rad[mask]
    rmin      = r_min*soft_length

    for i in range(num_halos):
        print('Computing profile %d/%d ... '%(i+1,num_halos))
        x,y  = ga.profile(pos-halopos[i:i+1,:],mass,BoxSize,rmin,rad[i]) 
        x = np.array(x); y = np.array(y)
        np.savetxt('aout/profile/'+str(snap_base)+'/f_'+str(stime)+'_gr_'+str(i)+'.txt',np.column_stack([x, y]))
        #fil = open('aout/profile/'+str(snap_base)+'/f_'+str(stime)+'_gr_'+str(i)+'.txt','w+')
        #for j in range(len(r)):
        #    fil.write('%.5f %.5f\n'%(r[j],rho[j]))
        #fil.close()
        
print("done!")




