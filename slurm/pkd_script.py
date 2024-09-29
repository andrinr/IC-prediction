from accuracy import classic_theta_switch,classic_replicas_switch
from PKDGRAV import add_analysis, ASSIGNMENT_ORDER

# accept command line arguments
import sys
if len(sys.argv) < 4:
    print("Usage: python3 params.py <seed> <output name> <nGrid>")
    sys.exit(1)

achOutName = sys.argv[2]

# Initial Condition
dBoxSize        = 60          # Mpc/h
nGrid           = 256           # Simulation has nGrid^3 particles
iLPT            = 2             # LPT order for IC
iSeed 			= int(sys.argv[1]) + 1000 	# Seed
dRedFrom        = 49            # Starting redshift

# Cosmology
achTfFile       = "euclid_z0_transfer_combined.dat"
h               = 0.67
dOmega0         = 0.32
dLambda         = 0.68
dSigma8         = 0.83
dSpectral       = 0.96

iStartStep      = 0
nSteps          = 100
dRedTo          = 0.0

# Cosmological Simulation
bComove         = True          # Use comoving coordinates
bPeriodic       = True          # with a periodic box
bEwald          = True          # enable Ewald periodic boundaries

# Logging/Output
iOutInterval    = 20
#iCheckInterval = 5
bDoDensity      = False
bVDetails       = True
bWriteIC        = True         # Write initial conditions

bOverwrite      = True
bParaRead       = True          # Read in parallel
bParaWrite      = False         # Write in parallel (does not work on all file systems)
#nParaRead      = 8             # Limit number of simultaneous readers to this
#nParaWrite     = 8             # Limit number of simultaneous writers to this

# Accuracy Parameters
bEpsAccStep     = True          # Choose eps/a timestep criteria
dTheta          = classic_theta_switch()        # 0.40, 0.55, 0.70 switch
nReplicas       = classic_replicas_switch()     # 1 if theta > 0.52 otherwise 2

# Memory and performance
bMemUnordered   = True          # iOrder replaced by potential and group id
bNewKDK         = True          # No accelerations in the particle, dual tree possible

class MassGrid:
    grid = 0
    order = ASSIGNMENT_ORDER.PCS
    def __init__(self,name,grid,order=ASSIGNMENT_ORDER.PCS):
        self.name = name
        self.grid = grid
        self.order = order
    def __call__(self,msr,step,time,**kwargs):
        if step % 20 == 0:
            print('calculating density grid')
            msr.grid_create(self.grid)
            msr.assign_mass(order=self.order)
            msr.grid_write(f'{self.name}.{step:05d}')
            msr.grid_delete()
    def ephemeral(self,msr,**kwargs):
        return msr.grid_ephemeral(self.grid)

add_analysis(MassGrid(sys.argv[3], nGrid))