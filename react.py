import numpy as np

class Particle(object):
    def __init__(self, species=None, mass=1.0, radius=1.0):
        self.species = species
        self.mass = mass
        self.radius = radius
        self.time = 0.0
        self.position = np.zeros(3)
        self.velocity = np.zeros(3)

class Reaction(object):
    def __init__(self, species_in, species_out, probability, time):
        self.species_in  = species_in
        self.species_out = species_out
        self.probability = probability
        self.time = time
        self.n_in = len(species_in)

speciesList = ['A', 'B', 'C']
spec_mass = {'A': 1.0, 'B': 2.0, 'C': 3.0}
spec_radius = {'A': 1.0, 'B': 1.26, 'C': 1.44} # r ~ mass^(1/3)

reactionSet = []
reactionSet.append(Reaction(['A', 'B'], ['C'],  1.0, None))
reactionSet.append(Reaction(['C'], ['A', 'B'], None, 2.0 ))

N = 100
Niter = 10
dt = 1.0
xhi = 100.0
yhi = 100.0
zhi = 100.0
vhi = 10.0

particleSet = []

def create_species_particle(s):
    # given a species s, create particle
    m = spec_mass[s]
    r = spec_radius[s]
    p = Particle(s, m, r)
    return p

def create_random_particle():
    s = np.random.choice(speciesList)
    p = create_species_particle(s)
    p.position = np.random.random_sample(3) * np.array([xi, yhi, zhi])
    p.velocity = np.random.random_sample(3) * vhi
    return p

# Detect colliding particles and yield the colliding set
def get_collisions(pset):
    for i in range(N):
        pi = pset[i]
        for j in range(N):
            if j == i:
                continue
            pj = pset[j]
            rij = pi.radius+pj.radius
            dij = np.linalg.norm(pi.position-pj.position)
            if dij <= rij:
                # Collision, yield the indices
                yield(i, j)

def kin_2_to_1(pc, pa, pb):
    # For a 2-to-1 collision A + B => C,
    # calculate the position and velocity of C
    momentum = pa.mass*pa.velocity + pb.mass*pb.velocity
    pc.velocity = momentum/pc.mass
    pc.position = 0.5*(pa.position + pb.position)

def kin_1_to_2(pc, pa, pb):
    # For a 1-to-2 decay C => A + B,
    # calculate the position and velocities of A, B
    momentum = pc.mass*pc.velocity
    energy   = 0.5*pc.mass*(pc.velocity**2)
    # TODO

def kin_1_to_1(pc, pa):
    # For a 1-to-1 decay C => A,
    # calculate the position and velocity of A
    pa.position = pc.position
    pa.velocity = pc.velocity

def calculate_out_kinematics(pinlist, poutlist):
    # Given lists of incoming and outgoing particles,
    # calculate the final state kinematics
    if len(pinlist) == 1 and len(poutlist) == 1:
        kin_1_to_1(pinlist[0], poutlist[0])
        return
    elif len(pinlist) == 1 and len(poutlist) == 2:
        kin_1_to_2(pinlist[0], poutlist[0], poutlist[1])
        return
    elif len(pinlist) == 2 and len(poutlist) == 1:
        kin_2_to_1(poutlist[0], pinlist[0], pinlist[1])
        return
    else:
        return

def get_outgoing_particles(pin, r):
    pout = []
    for s in r.species_out:
        p = create_species_particle(s)
        pout.append(p)
    # Calculate outgoing kinematics
    calculate_out_kinematics(pin, pout)
    return pout
    
# Create a bunch of random particles
for i in range(N):
    p = create_random_particle()
    particleSet.append(p)
    
# Iterate in time
for ti in range(Niter):
    # Check for expired timers and react
    for i in range(N):
        pi = particleSet[i]
        for r in reactionSet:
            if r.n_in == 1 and r.species_in[0] == pi.species:
                if pi.time >= r.time:
                    # Pop particle pi and create particles for species_out
                    pin = [particleSet.pop(i)]
                    pout = get_outgoing_particles(pin, r)
                    # Append outgoing particles to particleSet
                    for p in pout:
                        particleSet.append(p)
                    break # reaction loop
    
    # Check for collisions and react
    for i, j in get_collisions(particleSet):
        # we know i != j
        if i < j:
            pj = particleSet.pop(j)
            pi = particleSet.pop(i)
        else:
            pi = particleSet.pop(i)
            pj = particleSet.pop(j)
        sij = set([pi.species, pj.species])
        for r in reactionSet:
            if set(r.species_in) == sij:
                pin = [pi, pj]
                pout = get_outgoing_particles(pin, r)
                # Append outgoing particles to particleSet
                for p in pout:
                    particleSet.append(p)
                break # reaction loop
    
    # Advance in time
    for i in range(N):
        particleSet[i].position += particleSet[i].velocity*dt
        particleSet[i].time += dt
