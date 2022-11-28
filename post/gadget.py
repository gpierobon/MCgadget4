import os,sys
import errno
import yt
import h5py as h5
import numpy as np
import numba

def check_folder(foldname,snap_base):
    try:
        os.mkdir('aout/'+str(foldname)+'/'+str(snap_base))
    except OSError as exc:
        if exc.errno != errno.EEXIST:
            raise
        pass

def set_thres(vthres,vthres2,fil_flag=False):
    filam      = fil_flag
    if fil_flag == True:
        threshold  = -0.05       # Defaulted to almost average density   
        thresh2    = -vthres2    # Input
        print("Finding filaments volume and density as function of redshift for values %.2f < delta < -0.05 ..."%(-vthres2))
    else:
        threshold  = -vthres     # Input 
        thresh2    = -1.0        # Defaulted to zero energy  
        print("Finding voids volume and density as function of redshift for values delta < %.2f ..."%(-vthres))
    return vthres, vthres2, fil_flag

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

def to_numpy(dr,dp):
    adr  = np.array(dr).astype('float32')
    ares = np.array(dp).astype('float32')
    return adr, ares


def random_disp():
    dx = np.random.uniform(0.25,1)
    dy = np.random.uniform(0.25,1)
    dz = np.random.uniform(0.25,1)
    dr = np.sqrt(dx**2+dy**2+dz**2)
    return dx,dy,dz,dr


#@numba.njit()
def density_gradient(delta,N):  # Only North-East-Front octant for now
    drli = []
    dpli = []

    size = len(delta)*4*2*1e-9
    print("Saving %.5f GBs"%size)

    for idx in range(len(delta)):

        dx,dy,dz,dr = random_disp() 
        res = CIC(idx,dx,dy,dz,N)
        drli.append(dr)
        dpli.append(res)

    return drli,dpli
        

@numba.njit()
def CIC(idx,dx,dy,dz,N):
    out0, out2, x0, x1 = idx2vec(idx,N)
    xyz = idx
    Xyz = out0
    xYz = out2
    xyZ = idx + N*N
    if x1 != N-1: XYz = Xyz + N
    else:         XYz = xYz + 1
    if x0 != N-1: XyZ = xyZ + 1
    else:         XyZ = xyZ + N
    if x1 != N-1: xYZ = xyZ + N
    else:         xYZ = xYz +N*N
    if x1 != N-1: XYZ = Xyz + N + N*N 
    else:         XYZ = xYz + 1 + N*N

    res = xyz * ((1.-dx) * (1.-dy) * (1.-dz)) +\
          Xyz * (dx      * (1.-dy) * (1.-dz)) +\
          xYz * ((1.-dx) * dy      * (1.-dz)) +\
          xyZ * (dx      * dy      * (1.-dz)) +\
          XYz * ((1.-dx) * (1.-dy) * dz     ) +\
          XyZ * (dx      * (1.-dy) * dz     ) +\
          xYZ * ((1.-dx) *  dy     * dz     ) +\
          XYZ * (dx      *  dy     * dz     )

    return res


@numba.njit()
def idx2vec(idx,N):
    out0 = 0
    out2 = 0
    tmp = idx/N
    S   = N*N

    x2  = tmp/N
    x1  = tmp - x2*N
    x0  = idx - tmp*N

    if x0 == 0:
        out0 = idx + 1
    else:
        if x0 == N-1:
            out0 = idx - N + 1
        else:
            out0 = idx + 1
    if x1 == 0:
        out2 = idx + N
    else:
        if x1 == N-1:
            out2 = idx - S + N
        else:
            out2 = idx + N

    return out0,out2,x0,x1

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

