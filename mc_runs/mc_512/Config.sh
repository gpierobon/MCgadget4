PERIODIC                                     # enables periodic boundary condistions
NTYPES=2                                     # number of particle types 

SELFGRAVITY

PMGRID=512                                   # basic mesh size for TreePM calculations
NSOFTCLASSES=2                               # number of different softening classes

POSITIONS_IN_32BIT                           # if set, use 32-integers for positions  (default for single precision)
IDS_32BIT                                    # selects 32-bit IDs for internal storage (default)
POWERSPEC_ON_OUTPUT                          # computes a matter power spectrum when the code writes a snapshot output

FOF                                          # enable FoF output
FOF_PRIMARY_LINK_TYPES=2                     # bitmask, 2^type for the primary dark matter type
FOF_GROUP_MIN_LEN=32                         # minimum group length (default is 32)
FOF_LINKLENGTH=0.2                           # linking length for FoF (default=0.2)

SUBFIND
