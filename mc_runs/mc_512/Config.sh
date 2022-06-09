PERIODIC                                     # enables periodic boundary condistions
NTYPES=2                                     # number of particle types 

SELFGRAVITY

PMGRID=512                                   # basic mesh size for TreePM calculations
NSOFTCLASSES=2                               # number of different softening classes

POSITIONS_IN_32BIT                           # if set, use 32-integers for positions  (default for single precision)
IDS_32BIT                                    # selects 32-bit IDs for internal storage (default)
POWERSPEC_ON_OUTPUT                          # computes a matter power spectrum when the code writes a snapshot output
