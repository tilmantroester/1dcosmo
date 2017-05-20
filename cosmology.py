import numpy as np

pi = np.pi

EULER_INTEGRATOR = 0
LEAPFROG_INTEGRATOR = 1

DKD_LEAPFROG = 0
KDK_LEAPFROG = 0

DIRECT_FORCE_CALCULATION = 0
PM_FORCE_CALCULATION = 1

REAL_SPACE_PM = 0

DT_TIMESTEPS = 0

NO_FORCE_BOUNDARY_TERM = 0
FORCE_PERIODIC_BOUNDARY_TERM = 1

class ParticleDistribution:
    def __init__(self, n_particle, n_grid, box_size, particle_mass=1, 
                       integrator=EULER_INTEGRATOR,
                       leapfrog_type=DKD_LEAPFROG,
                       force_calculation=DIRECT_FORCE_CALCULATION,
                       pm_type=REAL_SPACE_PM,
                       force_boundary_term=FORCE_PERIODIC_BOUNDARY_TERM):
        self.n_particle = n_particle
        self.n_grid = n_grid
        self.L = box_size

        self.x = np.empty(n_particle)
        self.v = np.empty(n_particle)
        self.f = np.empty(n_particle)
        self.m = particle_mass
        self.t = 0

        self.integrator = integrator
        self.leapfrog_type = leapfrog_type
        self.force_calculation = force_calculation
        self.pm_type = pm_type

        self.force_boundary_term = force_boundary_term

        self.density = np.zeros(n_grid)
        if self.force_calculation == PM_FORCE_CALCULATION:
            self.cum_density = np.empty(n_grid)

    def set_initial_conditions(self, x0, v0):
        self.x = np.copy(x0)
        self.v = np.copy(v0)

    def run(self, t, time_unit=DT_TIMESTEPS, particles=True, density=True):
        if particles: particle_snapshots = np.empty((len(t), 2, self.n_particle))
        if density: density_snapshots = np.empty((len(t), self.n_grid))

        timesteps = t[1:]-t[:-1]
        for i, dt in enumerate(timesteps):
            if particles:
                particle_snapshots[i,0] = self.x
                particle_snapshots[i,1] = self.v
            if density:
                if self.force_calculation != PM_FORCE_CALCULATION:
                    self.calculate_density()
                density_snapshots[i] = self.density
            self.evolve(dt, time_unit)

        if particles and density:
            return particle_snapshots, density_snapshots
        elif particles:
            return particle_snapshots
        elif density:
            return density_snapshots

    def evolve(self, dt, time_unit):
        if time_unit == DT_TIMESTEPS:
            if self.integrator == EULER_INTEGRATOR:
                self.evolve_dt_euler(dt)
            elif self.integrator == LEAPFROG_INTEGRATOR:
                self.evolve_dt_leapfrog(dt)

        self.collapse_periodic_images()

        self.t += dt

    def evolve_dt_euler(self, dt):
        self.calculate_forces()

        self.x += self.v*dt + 0.5*self.f*dt**2
        self.v += self.f*dt

    def evolve_dt_leapfrog(self, dt):
        if self.leapfrog_type == DKD_LEAPFROG:
            #Drift
            self.x += self.v*dt/2
            #Kick
            self.calculate_forces()
            self.v += self.f*dt
            #Drift
            self.x += self.v*dt/2
        elif self.leapfrog_type == KDK_LEAPFROG:
            #Kick
            self.calculate_forces()
            self.v += self.f*dt/2
            #Drift
            self.x += self.v*dt
            #Kick
            self.calculate_forces()
            self.v += self.f*dt/2

    def calculate_forces(self):
        if self.force_calculation == DIRECT_FORCE_CALCULATION:
            sort_idx = np.argsort(self.x)
            self.x = self.x[sort_idx]
            self.v = self.v[sort_idx]

            #Difference of particles to the right and left
            delta_N = self.n_particle-1-2*np.arange(0, self.n_particle)
            self.f = self.m*delta_N

        elif self.force_calculation == PM_FORCE_CALCULATION:
            if self.pm_type == REAL_SPACE_PM:
                self.calculate_density()
                grid_idx = np.asarray((self.x/self.L + 0.5)*self.n_grid - 0.5, dtype=int)
                self.cum_density = np.cumsum(self.density)

                upper_edge_particle_idx = grid_idx == self.n_grid-1
                self.f[upper_edge_particle_idx] = -self.cum_density[-2]
                lower_edge_particle_idx = grid_idx <= 0
                self.f[lower_edge_particle_idx] = self.cum_density[-1]-self.cum_density[1]
                bulk_particle_idx = np.logical_and(grid_idx>0, grid_idx<self.n_grid-1)
                self.f[bulk_particle_idx] = self.cum_density[-1]-self.cum_density[grid_idx[bulk_particle_idx]+1]-self.cum_density[grid_idx[bulk_particle_idx]-1]

        if self.force_boundary_term == FORCE_PERIODIC_BOUNDARY_TERM:
            self.f += -2*np.mean(self.x)

    def calculate_density(self):
        grid_idx = np.asarray((self.x/self.L + 0.5)*self.n_grid - 0.5, dtype=int)
        self.density = np.zeros(self.n_grid)
        for i in range(self.n_particle):
            self.density[grid_idx[i]] += self.m

    def collapse_periodic_images(self):
        out_of_lower_bound_idx = self.x <= -self.L/2
        self.x[out_of_lower_bound_idx] += self.L
        out_of_upper_bound_idx = self.x > self.L/2
        self.x[out_of_upper_bound_idx] -= self.L



