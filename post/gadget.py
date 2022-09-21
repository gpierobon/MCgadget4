import os,sys
import errno
import yt
import h5py as h5
import numpy as np

def check_folder(foldname,snap_base):
    try:
        os.mkdir('aout/'+str(foldname)+'/'+str(snap_base))
    except OSError as exc:
        if exc.errno != errno.EEXIST:
            raise
        pass

def get_info(f):
    """
    Prints info for a given snapshot
    """
    filename = f+'.hdf5'
    fi = h5.File(filename, 'r')
    print('File: ',fi, '\n')
    print("Redshift: %.1f"%fi['Header'].attrs['Redshift'])
    print("NumParts: %d^3"%int(np.ceil(fi['Header'].attrs['NumPart_Total'][1]**(1/3))))
    print("Box Size: %.5f pc"%fi['Header'].attrs['BoxSize'])


def load_halos(f,fof=True,radius='R200',additional=False,verbose=True):
    """
    Loads fof data for a given fof tab
    Used for profiles of MC halos or general properties 
    """
    filename = str(f)+'.hdf5'
    
    if verbose:
        if fof:
            print('Extracting FOF data from %s'%filename)
        else:
            print('Extracting SubHalo data from %s'%filename)
            
    fi = h5.File(filename, 'r')
    head = dict(fi['Header'].attrs)
    z = head['Redshift']
    a = 1./(1+z)
    pos   = []
    vel   = []
    mass  = []
    rad   = []
    size  = []
    out   = []
    
    if fof:
        pos  += [np.array(fi['Group/GroupPos'])]
        vel  += [np.array(fi['Group/GroupVel'])*np.sqrt(a)]
        mass += [np.array(fi['Group/GroupMass'])]
        if radius == 'R200':
            rad += [np.array(fi['Group/Group_R_Crit200'])]
        elif radius == 'R500':
            rad += [np.array(fi['Group/Group_R_Crit500'])]
        elif radius == 'RMean':
            rad += [np.array(fi['Group/Group_R_Mean200'])]
        elif radius == 'TopHat':
            rad += [np.array(fi['Group/Group_R_TopHat200'])]
        else:
            raise ValueError('Selected radius is unknown')

        size += [np.array(fi['Group/GroupLen'])]

    else:

        pos   += [np.array(fi['Subhalo/SubhaloCM'])]
        vel   += [np.array(fi['Subhalo/SubhaloVel'])*np.sqrt(a)]
        mass  += [np.array(fi['Subhalo/SubhaloMass'])]
        rad   += [np.array(fi['Subhalo/SubhaloHalfmassRad'])]
        size  += [np.array(fi['Subhalo/SubhaloLen'])] 
       
        if additional:
            vdisp = []
            spin  = []
            vdisp += [np.array(fi['Subhalo/SubhaloVelDisp'])]
            spin  += [np.array(fi['Subhalo/SubhaloSpin'])] 
    

    out  += [np.concatenate(pos,axis=1)]
    out  += [np.concatenate(vel,axis=1)]
    out  += [np.concatenate(mass)]
    out  += [np.concatenate(rad)]
    out  += [np.concatenate(size)]

    if additional:
        out += [np.concatenate(vdisp)]
        out += [np.concatenate(spin)]

    if verbose:
        if additional:
            print('%d halos loaded: data is stored in header,pos,vel,mass,radius,size,vdisp,spin'%len(pos[0]))
        else:
            print('%d halos loaded: data is stored in header,pos,vel,mass,radius,size'%len(pos[0]))
    
    return head,out 


def load_particles(f,verbose=True):
    """
    Loads particle data for a given snapshot
    Usage:
    pos,vel,mass,ID = load_particles('/path/to/snap',verbose=True)

    """
    filename = str(f)+'.hdf5'
    
    if verbose:
        print('Extracting particle data from %s'%filename)
            
    fi = h5.File(filename, 'r')
    head = dict(fi['Header'].attrs)
    z = head['Redshift']
    a = 1./(1+z)
    mtab = head['MassTable']
    nparts = head['NumPart_Total'][1]

    pos  = np.zeros((3,np.sum(nparts)),dtype=np.float32)
    vel  = np.zeros((3,np.sum(nparts)),dtype=np.float32)
    mass = np.zeros(np.sum(nparts),dtype=np.float32)
    ID   = np.zeros(np.sum(nparts),dtype=np.uint32)
    out  = []

    pos = np.array(fi['PartType1/Coordinates'])
    vel = np.array(fi['PartType1/Velocities'])*np.sqrt(a)

    if mtab[1] == 0.:
        mass = np.array(fi['PartType1/Masses'])
    else:
        mass = np.full(nparts,mtab[1])

    ID = np.array(fi['PartType1/ParticleIDs'])
     
    out += [pos]
    out += [vel]
    out += [mass]
    out += [ID]
    if verbose:
        print('%d particles loaded: data is stored in header,pos,vel,mass,ID'%nparts)

    return head,out 


def profile(x,mass,L,rmin,rad,nbins=50):
    '''
    Computes profiles 
    '''
    nparts = x.shape[0]
    x[x >= 0.5*L] -= L
    x[x < -0.5*L] -= L
    
    r    = np.sqrt(np.sum(x.T**2,axis=0))
    bins = np.geomspace(rmin,rad,nbins)

    bvol = 4./3 * np.pi * (bins[1:]**3 - bins[:-1]**3)
    hist_mass,hbins = np.histogram(r,bins=bins,weights=mass)
    
    r_out = 0.5*(bins[1:]+bins[:-1])
    rho_out = hist_mass/bvol
    
    return r_out/rad, rho_out


def vol_render(de,res,bmin,bmax,snap_base,j):
    data = dict(density = (de, "g/cm**3"))
    ds = yt.load_uniform_grid(data, de.shape, length_unit="pc")
    sc = yt.create_scene(ds, field=("density"))
    sc.camera.resolution = (res, res)
    sc.camera.focus = ds.arr([0.3, 0.3, 0.3], "unitary")
    source = sc[0]
    source.tfh.set_bounds((bmin, bmax))
    sc.camera.position = ds.arr([0, 0, 0], "unitary")
    sc.render()
    sc.save(f'aout/render/'+str(snap_base)+f'/shell_{j}.png', sigma_clip=4)

