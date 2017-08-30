import numpy as np
import numba
import math
import os

import scipy.integrate

import matplotlib.pyplot as plt
import matplotlib.animation

import utils

pi = np.pi

EULER_INTEGRATOR = 0
LEAPFROG_INTEGRATOR = 1
ZELDOVICH_INTEGRATOR = 2

DKD_LEAPFROG = 0
KDK_LEAPFROG = 1

EUCLIDEAN_DRIFT = 0
SYMPLECTIC_DRIFT = 1
EUCLIDEAN_KICK = 1
FLRW_KICK = 2
SYMPLECTIC_KICK = 3

DIRECT_FORCE_CALCULATION = 0
PM_FORCE_CALCULATION = 1

NGP_MESH_INTERPOLATION = 0
CIC_MESH_INTERPOLATION = 1

REAL_SPACE_PM = 0
FFT_PM = 1

DT_TIMESTEPS = 0
DLAMBDA_TIMESTEPS = 1

NO_FORCE_BOUNDARY_TERM = 0
FORCE_PERIODIC_BOUNDARY_TERM = 1

DENSITY_SCALING_1D = 0
DENSITY_SCALING_3D = 1

class ParticleDistribution:
    def __init__(self, n_particle, n_grid, box_size, particle_mass=1, 
                       integrator=EULER_INTEGRATOR,
                       time_unit=DT_TIMESTEPS,
                       leapfrog_type=DKD_LEAPFROG,
                       leapfrog_drift_type=EUCLIDEAN_DRIFT,
                       leapfrog_kick_type=EUCLIDEAN_KICK,
                       force_calculation=DIRECT_FORCE_CALCULATION,
                       pm_type=REAL_SPACE_PM,
                       mesh_interpolation=NGP_MESH_INTERPOLATION,
                       Omega_m=0.3, Omega_L=0.7, h=None,
                       beta=0.5,
                       density_scaling=DENSITY_SCALING_3D,
                       force_boundary_term=FORCE_PERIODIC_BOUNDARY_TERM,
                       t_min=0.0, a_min=0.0, precision=np.float32):
        self.n_particle = n_particle
        self.n_grid = n_grid
        self.L = box_size

        self.x = np.zeros(n_particle, dtype=precision)
        self.v = np.zeros(n_particle, dtype=precision)
        self.f = np.zeros(n_particle, dtype=precision)
        self.m = particle_mass
        
        self.beta = beta
        self.t = t_min
        self.a = a_min
        self.l = a_min**self.beta

        self.integrator = integrator
        self.time_unit = time_unit
        self.leapfrog_type = leapfrog_type
        self.leapfrog_drift_type = leapfrog_drift_type
        self.leapfrog_kick_type = leapfrog_kick_type
        self.force_calculation = force_calculation
        self.pm_type = pm_type

        self.force_boundary_term = force_boundary_term

        self.mesh_interpolation = mesh_interpolation

        self.Omega_m = Omega_m
        self.Omega_L = Omega_L
        self.h = h
        self.H0 = 1
        self.density_scaling = density_scaling
        if self.density_scaling == DENSITY_SCALING_1D:
            self.Hubble = self.Hubble_1D
            self.scale_factor = self.scale_factor_1D
            self.t_of_a = self.t_of_a_1D
            self.a_ddot = self.a_ddot_1D
            raise Warning("1D density scales messes with growth rate.")
        elif self.density_scaling == DENSITY_SCALING_3D:
            self.Hubble = self.Hubble_3D
            self.scale_factor = self.scale_factor_3D
            self.t_of_a = self.t_of_a_3D
            self.a_ddot = self.a_ddot_3D

        self.density = np.zeros(n_grid, dtype=precision)

        if h is None:
            self.G_prime = 1.0
        else:
            # Cosmology
            if self.Omega_m + self.Omega_L != 1.0:
                raise NotImplementedError("Only flat cosmologies are supported at the moment but Omega_m + Omega_L = {}".format(self.Omega_m + self.Omega_L))


            # Convert to dimensionless units
            self.c = 2.99792458e8 # m s^-1
            self.Mpc_m = 3.085677581e22 # 1 Mpc / m
            self.H0_s = 100/self.Mpc_m*1.0e3 # h s^-1
            self.H0_Gyr = self.H0_s*365*24*3600*1.0e9

            # Convert to units such that L = 1 and t(a=1) = 1
            self.L0 = box_size
            self.H0 = self.H0_Gyr
            self.T0 = self.t_of_a(1.0)
            self.H0 = self.H0*self.T0
            self.T0 = self.t_of_a(1.0)

            self.L /= self.L0
            self.m = 1.0

            self.G_prime_1D = self.H0**2/4*self.Omega_m/self.n_particle*self.L
            self.G_prime_3D = 3*self.H0**2/4*self.Omega_m/self.n_particle*self.L
            if self.density_scaling == DENSITY_SCALING_1D:
                self.G_prime = self.G_prime_1D
            elif self.density_scaling == DENSITY_SCALING_3D:
                self.G_prime = self.G_prime_3D

            # self.T0 = 1/self.H0_s # h^-1 s 
            # # Note: H_0 = 1 in units of T0
            # self.H0 = 1 # T0
            # self.L0 = self.c/(100*1.0e3) # c/H_0 in Mpc/h
            # self.G_Mpc_km2_s2_M_sol = 4.302e-9 #Mpc km^2 s^-2 M_sol^-1
            # self.G_h_L03_T02_M_sol = 4.302e-9/(self.Mpc_m*1.0e-3)**2/self.L0**3*self.T0**2 #h L0^3 T0^-2 M_sol^-1
            # self.rho_c_h_M_sol_L03 = 3*1/(8*pi*self.G_h_L03_T02_M_sol) # h^-1 M_sol L0^-3
            
            # self.rho_c_h2_M_sol_Mpc3 = 3*1.0e4/(8*pi*self.G_Mpc_km2_s2_M_sol) # h^2 M_sol Mpc^-3
            # self.m_h_M_sol_Mpc2 = self.rho_c_h2_M_sol_Mpc3*self.Omega_m*self.L/self.n_particle # h M_sol Mpc^-2
            
            # self.L = self.L/self.L0
            # self.x = self.x/self.L0
            # self.v = self.v/self.L0*self.T0
            # self.m_h_M_sol_L02 = self.rho_c_h_M_sol_L03*self.Omega_m*self.L/self.n_particle # h^-1 M_sol L0^-2

            # self.G = self.G_h_L03_T02_M_sol
            # self.rho_c = self.rho_c_h_M_sol_L03
            # self.m = self.m_h_M_sol_L02*6
                

    def set_initial_conditions(self, x0, v0):
        self.x = np.copy(x0)
        self.v = np.copy(v0)
        if self.h is not None:
            self.x = self.x/self.L0
            self.v = self.v/self.L0*self.T0

    def create_initial_conditions_CDM(self, P, a0):
        self.linear_powerspectrum = lambda k: P(k/self.L0)/self.L0

        d0, d0k, k = utils.create_Gaussian_field_1d(self.linear_powerspectrum, self.n_grid, self.L, output_FT=True)
        
        u0k = np.zeros_like(d0k)
        u0k[k!=0] = 1j/k[k!=0]*d0k[k!=0]
        u0k[k==0] = 0
        u0 = np.fft.irfft(u0k)
        if self.integrator == ZELDOVICH_INTEGRATOR:
            self.v0 = u0

        self.evolve_zeldovich(u0, a0, check_shell_crossing=True)
        
        self.collapse_periodic_images()
        self.calculate_density()

    def create_initial_conditions_density(self, d, a0):
        d0k = np.fft.rfft(d)
        k = np.fft.rfftfreq(self.n_grid)*2*pi/self.L*self.n_grid
        
        u0k = np.zeros_like(d0k)
        u0k[k!=0] = 1j/k[k!=0]*d0k[k!=0]
        u0k[k==0] = 0
        u0 = np.fft.irfft(u0k)
        if self.integrator == ZELDOVICH_INTEGRATOR:
            self.v0 = u0

        self.evolve_zeldovich(u0, a0, check_shell_crossing=True)
        
        self.calculate_density()
        

    def run(self, t, snapshots=[], verbose=False):
        print("Running simulation")
        print("Time stepping: {}".format("dt" if self.time_unit==DT_TIMESTEPS else "dl, l = a^{}".format(self.beta)))
        #print("Start t: {:.2e}, end t: {:.2e}".format(*(t[0, t[-1]) if time_unit==DT_TIMESTEPS else *(self.t)]))

        if self.time_unit == DT_TIMESTEPS:
            if self.h is not None:
                # Get t in units T0 (from Gyr)
                t = t/self.T0
            self.t = t[0]
            self.a = self.scale_factor(self.t)
            self.l = self.a**self.beta
        elif self.time_unit == DLAMBDA_TIMESTEPS:
            self.l = t[0]
            self.a = self.l**(1/self.beta)
            self.t = self.t_of_a(self.a)

        for snapshot in snapshots:
            if np.any(snapshot["timestep_idx"] == 0):
                #Initial conditions
                snapshot["snapshots"].take_snapshot(self, verbose)

        timesteps = t[1:]-t[:-1]
        for i, dt in enumerate(timesteps):
            if verbose:
                if self.time_unit ==  DT_TIMESTEPS:
                    if self.h is not None:
                        print("t = {:.2e}, dt = {:.2e},  a = {:.2f}, l = {:.2e}".format(self.t, dt, self.a, self.l))
                    else:
                        print("t = {:.2e}, dt = {:.2e}".format(self.t, dt))
                elif self.time_unit ==  DLAMBDA_TIMESTEPS:
                    print("l = {:.2e}, dl = {:.2e}, a = {:.2f}, t = {:.2e}".format(self.l, dt, self.a, self.t))
            self.evolve(dt)

            for snapshot in snapshots:
                if np.any(snapshot["timestep_idx"] == i+1):
                    snapshot["snapshots"].take_snapshot(self, verbose)


    def evolve(self, dt):
        if self.time_unit == DT_TIMESTEPS:
            if self.integrator == EULER_INTEGRATOR:
                self.evolve_dt_euler(dt)
            elif self.integrator == LEAPFROG_INTEGRATOR:
                self.evolve_dt_leapfrog(dt)
            elif self.integrator == ZELDOVICH_INTEGRATOR:
                a = self.scale_factor(self.t+dt)
                self.evolve_zeldovich(self.v0, self.a)
            self.t += dt
            self.a = self.scale_factor(self.t)
            self.l = self.a**self.beta

        elif self.time_unit == DLAMBDA_TIMESTEPS:
            if self.integrator == LEAPFROG_INTEGRATOR:
                self.evolve_dl_leapfrog(dt)
            elif self.integrator == ZELDOVICH_INTEGRATOR:
                a = (self.l+dt)**(1/self.beta)
                self.evolve_zeldovich(self.v0, self.a)

            self.l += dt
            self.a = self.l**(1/self.beta)
            self.t = self.t_of_a(self.a)

    def evolve_dt_euler(self, dt):
        self.calculate_forces()

        self.x += self.v*dt + 0.5*self.f*dt**2
        self.v += self.f*dt

    def evolve_dt_leapfrog(self, dt):
        if self.leapfrog_type == DKD_LEAPFROG:
            self.drift_dt(dt/2)
            self.calculate_forces()
            self.kick_dt(dt)
            self.drift_dt(dt/2)
        elif self.leapfrog_type == KDK_LEAPFROG:
            self.calculate_forces()
            self.kick_dt(dt/2)
            self.drift_dt(dt)
            self.calculate_forces()
            self.kick_dt(dt/2)

    def evolve_dl_leapfrog(self, dl):
        if self.leapfrog_type == DKD_LEAPFROG:
            self.drift_dl(dl/2)
            self.calculate_forces()
            self.kick_dl(dl)
            self.drift_dl(dl/2)
        elif self.leapfrog_type == KDK_LEAPFROG:
            raise NotImplementedError("KDK not implemented for lambda timesteps.")

    def drift_dt(self, dt):
        if self.leapfrog_drift_type == EUCLIDEAN_DRIFT:
            self.x += self.v*dt
        elif self.leapfrog_drift_type == SYMPLECTIC_DRIFT:
            a1 = self.scale_factor(self.t)
            a12 = self.scale_factor(self.t+dt/2)
            a2 = self.scale_factor(self.t+dt)
            A = 2/self.H0*(1/np.sqrt(a1) - 1/np.sqrt(a2))
            self.x += a12**2*self.v*A

    def kick_dt(self, dt):
        if self.leapfrog_kick_type == EUCLIDEAN_KICK:
            self.v += self.f*dt
        elif self.leapfrog_kick_type == FLRW_KICK:
            a = self.scale_factor(self.t+dt/2)
            H = self.Hubble(a)
            if self.density_scaling == DENSITY_SCALING_1D:
                kick = (self.v*(1-H*dt) + self.f/a*dt)/(1+H*dt)
                #kick = self.v*(1-2*H*dt) + 2*pi*self.G*self.f/a*dt
            elif self.density_scaling == DENSITY_SCALING_3D:
                kick = (self.v*(1-H*dt) + self.f/a**3*dt)/(1+H*dt)
                #kick = self.v*(1-2*H*dt) + self.f/a**3*dt
            self.v = kick
        elif self.leapfrog_kick_type == SYMPLECTIC_KICK:
            a1 = self.scale_factor(self.t)
            a2 = self.scale_factor(self.t+dt)
            A = 2/self.H0*(np.sqrt(a2) - np.sqrt(a1))
            p1 = a1**2*self.v
            p2 = p1 + A*self.f
            self.v = p2/a2**2

    def drift_dl(self, dl):
        self.x += self.v*dl

    def kick_dl(self, dl):
        beta = self.beta
        a = (self.l+dl/2)**(1/beta)
        l = a**beta
        H = self.Hubble(a)
        B = 1/l*(1/beta+1) + self.a_ddot(a)/(beta*a*l*H**2)
        if self.density_scaling == DENSITY_SCALING_1D:
            A = 1/(l**2*H**2)
        elif self.density_scaling == DENSITY_SCALING_3D:
            A = 1/(beta**2*a**3*l**2*H**2)
        self.v = (self.v*(1-B/2*dl) + A*self.f*dl)/(1+B/2*dl)

        
    def evolve_zeldovich(self, u0, a, check_shell_crossing=False):
        if self.time_unit == DT_TIMESTEPS:
            u = u0*self.dgrowth_rate_dt(a)
        elif self.time_unit == DLAMBDA_TIMESTEPS:
            u = u0*self.dgrowth_rate_dl(a)
        Psi = u0*self.growth_rate(a)

        self.x = self.L*(np.linspace(0, 1, self.n_particle, endpoint=False, dtype=self.x.dtype)+ 0.5/self.n_particle)
        self.v = grid_lookup_cic_numba(self.x, u, self.n_particle, self.n_grid, self.L)
        
        displacement = grid_lookup_cic_numba(self.x, Psi, self.n_particle, self.n_grid, self.L)
        self.x += displacement
        if check_shell_crossing:
            if not np.all(self.x[:-1] <= self.x[1:]):
                raise Warning("Shell-crossing during Zeldovich approximation for a={}".format(a))
        self.collapse_periodic_images()


    def calculate_forces(self):
        self.collapse_periodic_images()
        if self.force_calculation == DIRECT_FORCE_CALCULATION:
            sort_idx = np.argsort(self.x)
            self.x = self.x[sort_idx]
            self.v = self.v[sort_idx]

            #Difference of particles to the right and left
            delta_N = self.n_particle-1-2*np.arange(0, self.n_particle)
            self.f = self.m*delta_N
            if self.force_boundary_term == FORCE_PERIODIC_BOUNDARY_TERM:
                self.f += 2*self.m*self.n_particle/self.L*(self.x-np.mean(self.x))

        elif self.force_calculation == PM_FORCE_CALCULATION:
            if self.pm_type == REAL_SPACE_PM:
                self.calculate_density()
                tmp = np.cumsum(self.density/2.0)
                force_grid = np.cumsum(self.density/2.0)
                force_grid[1:] += tmp[:-1]
                force_grid = -2*force_grid + self.n_particle*self.m
                if self.force_boundary_term == FORCE_PERIODIC_BOUNDARY_TERM:
                    x_cell = (np.arange(0, self.n_grid)+0.5)*self.L/self.n_grid
                    x_cell_c = np.sum(self.density*x_cell)/np.sum(self.density)
                    force_grid += 2*self.m*self.n_particle/self.L*(x_cell-x_cell_c)
            elif self.pm_type == FFT_PM:
                self.calculate_density()
                dk = np.fft.rfft(self.density/np.mean(self.density) - 1)
                k = 2*pi/self.L*self.n_grid*np.fft.rfftfreq(self.n_grid)
                fk = np.zeros_like(dk)
                fk[k!=0] = 1j/k[k!=0]*2*dk[k!=0]/self.L*self.n_particle
                fk[k==0] = 0
                force_grid = np.fft.irfft(fk)

            if self.mesh_interpolation == NGP_MESH_INTERPOLATION:
                grid_idx = np.asarray(np.floor(self.x/self.L*self.n_grid), dtype=int)
                self.f = force_grid[grid_idx]
            elif self.mesh_interpolation == CIC_MESH_INTERPOLATION:
                self.f = grid_lookup_cic_numba(self.x, force_grid, self.n_particle, self.n_grid, self.L)

        self.f *= self.G_prime
        

    def collapse_periodic_images(self):
        self.x = np.mod(self.x, self.L)

    def calculate_density(self):
        if self.mesh_interpolation == NGP_MESH_INTERPOLATION:
            self.density = grid_create_ngp_numba(self.x, self.n_grid, self.L, self.m)
        elif self.mesh_interpolation == CIC_MESH_INTERPOLATION:
            self.density = grid_create_cic_numba(self.x, self.n_grid, self.L, self.m)
    
    def scale_factor_1D(self, t):
        if self.Omega_L > 0:
            X = self.H0*np.sqrt(self.Omega_L)*t
            a = self.Omega_m/self.Omega_L*np.sinh(0.5*X)**2
        else:
            a = self.H0**2*t**2/4
        return a

    def Hubble_1D(self, a):
        return self.H0*np.sqrt(self.Omega_m*a**-1 + self.Omega_L + (1-self.Omega_m-self.Omega_L)*a**-2)

    def t_of_a_1D(self, a):
        if self.Omega_L > 0:
            t = 2*np.arcsinh(np.sqrt(a*self.Omega_L/self.Omega_m))/(self.H0*np.sqrt(self.Omega_L))
        else:
            t = 2*np.sqrt(a)/self.H0
        return t

    def a_ddot_1D(self, a):
        return a*self.H0**2*(self.Omega_L - 1/2*self.Omega_m*a**-1)

    def scale_factor_3D(self, t):
        if self.Omega_L > 0:
            X = self.H0*np.sqrt(self.Omega_L)*t
            a = (self.Omega_m/self.Omega_L)**(1/3)*np.exp(-X)*(np.exp(3*X)-1)**(2/3)/2**(2/3)
        else:
            a = (3*self.H0/2*t)**(2/3)
        return a

    def Hubble_3D(self, a):
        return self.H0*np.sqrt(self.Omega_m*a**-3 + self.Omega_L + (1-self.Omega_m-self.Omega_L)*a**-2)

    def t_of_a_3D(self, a):
        OL = self.Omega_L
        Om = self.Omega_m
        if self.Omega_L > 0:
            t = (2*np.log(a**(3/2)*OL + np.sqrt(OL*(a**3*OL+Om))) - np.log(OL*Om))/(3*self.H0*np.sqrt(OL))
        else:
            t = 2/(3*self.H0)*a**(3/2)
        return t

    def a_ddot_3D(self, a):
        return a*self.H0**2*(self.Omega_L - 1/2*self.Omega_m*a**-3)

    def growth_rate(self, a):
        growth_function_intg = lambda a: 1.0/(a*self.Hubble(a)/self.H0)**3
        integral = scipy.integrate.quad(growth_function_intg, 0, a)[0]
        
        #A = 0.9959520 #= scipy.integrate.quad(growth_function_intg, 0, 1.0)[0]
        A = 1.0/( 5.0/2.0*self.Omega_m )       
        #A = 1.038641
        return 1.0/A * self.Hubble(a)/self.H0 * integral

    def dgrowth_rate_dt(self, a):
        growth_function_intg = lambda a: 1.0/(a*self.Hubble(a)/self.H0)**3
        integral = scipy.integrate.quad(growth_function_intg, 0, a)[0]
        A = 1.0/( 5.0/2.0*self.Omega_m )       
        #A = 1.038641
        return 1.0/A/self.H0 * ((self.a_ddot_3D(a)/a- self.Hubble(a)**2)*integral \
                        + a*self.Hubble(a)**2/(a*self.Hubble(a)/self.H0)**3)
    
    def dgrowth_rate_dl(self, a):
        l = a**self.beta
        growth_function_intg = lambda a: 1.0/(a*self.Hubble(a)/self.H0)**3
        integral = scipy.integrate.quad(growth_function_intg, 0, a)[0]
        
        #A = 0.9959520 #= scipy.integrate.quad(growth_function_intg, 0, 1.0)[0]
        A = 1.0/( 5.0/2.0*self.Omega_m )       
        #A = 1.038641
        return 1.0/A/self.H0 * (1/(self.beta*l*self.Hubble(a))*(self.a_ddot_3D(a)/a - self.Hubble(a)**2)*integral \
                        + self.Hubble(a)*a/(self.beta*l)/(a*self.Hubble(a)/self.H0)**3) 

    def growth_rate_3D(self, a):
        """Fitting function of Lahal 1991"""
        omega_m = self.Omega_m*a**(-3)/(self.Hubble(a)/self.H0)**2
        omega_L = self.Omega_L/(self.Hubble(a)/self.H0)**2
        
        g = 5.0/2.0*omega_m * 1.0/(omega_m**(4.0/7.0) - omega_L + (1+omega_m/2.0)*(1+omega_L/70.0))
        g0 = 5.0/2.0*self.Omega_m
        return g*a#/g0


@numba.jit(nopython=True)
def grid_create_ngp_numba(x, n_grid, L, m):
    density = np.zeros(n_grid, dtype=x.dtype)
    for i in range(len(x)):
        idx = math.floor(x[i]/L*n_grid)
        density[idx] += m
    return density

@numba.jit(nopython=True)
def grid_create_cic_numba(x, n_grid, L, m):
    density = np.zeros(n_grid, dtype=x.dtype)
    grid_spacing = L/n_grid
    for i in range(len(x)):
        idx = math.floor(x[i]/grid_spacing)
        x_idx = grid_spacing*(idx+0.5)
        r = (x_idx - x[i])/grid_spacing
        if r >= 0:
            density[idx] += (1-r)*m
            density[idx-1] += r*m
        else:
            density[idx] += (1+r)*m
            if idx < n_grid-1:
                density[idx+1] += -r*m
            else:
                density[0] += -r*m
    return density


@numba.jit(nopython=True)
def grid_lookup_cic_numba(x, force_grid, n_particle, n_grid, L):
    f = np.zeros(n_particle, dtype=x.dtype)
    grid_spacing = L/n_grid
    for i in range(len(x)):
        idx = math.floor(x[i]/grid_spacing)
        x_idx = grid_spacing*(idx+0.5)
        r = (x_idx - x[i])/grid_spacing
        if r >= 0:
            f[i] += (1-r)*force_grid[idx]
            f[i] += r*force_grid[idx-1]
        else:
            f[i] += (1+r)*force_grid[idx]
            if idx < n_grid-1:
                f[i] += -r*force_grid[idx+1]
            else:
                f[i] += -r*force_grid[0]
    return f

def find_halos_fof(x, l, L):
    halo_idx = np.zeros(x.size, dtype=np.int64)

    current_halo_idx = 0
    for i in range(1, x.size):
        if x[i]-x[i-1] > l:
            current_halo_idx += 1
        halo_idx[i] = current_halo_idx

    #Assign particles across the periodic boundary to first halo (idx=0)
    if x[0]+L - x[-1] <= l:
        halo_idx[-1] = 0
    for i in range(1, x.size):
        if x[-i]-x[-i-1] <= l:
            halo_idx[i] = 0
        else:
            break

    return halo_idx

def insert_snapshot_timesteps(t, snapshot_timesteps):
    duplicate_idx = np.argwhere(np.isin(snapshot_timesteps, t)).flatten()
    new_ss_timesteps = np.delete(snapshot_timesteps, duplicate_idx)
    new_t = np.insert(t, np.searchsorted(t, new_ss_timesteps), new_ss_timesteps)
    ss_timestep_idx = np.argwhere(np.isin(new_t, snapshot_timesteps)).flatten()

    return new_t, ss_timestep_idx

class ParticleSnapshots:
    def __init__(self, particles,
                       run_id=0,
                       n_snapshot=None,
                       phase_space=False, particle_downsampling=1,
                       density=False, density_downsampling=1,
                       power_spectrum=False, n_k_bin=None, k_min=None, k_max=None, logspaced=True, linear_powerspectrum=False,
                       phases=False,
                       halos=False,
                       keep_in_memory=True,
                       output_directory=None,
                       phase_space_filename_format="run{run_id}_L{L:.0f}_n{n_particle:.1e}_z{z:.3f}_phase_space",
                       density_filename_format="run{run_id}_L{L:.0f}_n{n_particle:.1e}_z{z:.3f}_density",
                       power_spectrum_filename_format="run{run_id}_L{L:.0f}_n{n_particle:.1e}_z{z:.3f}_power_spectrum",
                       file_format="npz"):
        if particles==None:
            raise RuntimeError("Require ParticleDistribution object for particles argument.")

        self.run_meta_info = {"id" : run_id, 
                              "L_box" : particles.L*particles.L0, 
                              "n_particle" : particles.n_particle}
        self.L_box =  self.run_meta_info["L_box"]

        self.keep_in_memory = keep_in_memory
        if self.keep_in_memory:
            self.n_snapshot = n_snapshot
        else:
            self.n_snapshot = 1
        self.current_snapshot_idx = 0

        self.output_directory = output_directory
        self.file_format = file_format
        self.phase_space_filename_format = phase_space_filename_format
        self.density_filename_format = density_filename_format
        self.power_spectrum_filename_format = power_spectrum_filename_format

        self.a = np.zeros(n_snapshot)
        self.t = np.zeros(n_snapshot)
        self.l = np.zeros(n_snapshot)
        self.beta = np.zeros(n_snapshot)
        self.Hubble = np.zeros(n_snapshot)
        self.growth_rate = np.zeros(n_snapshot)

        if phase_space:
            self.particle_downsampling = particle_downsampling
            self.phase_space = np.zeros((self.n_snapshot, 2, particles.n_particle // particle_downsampling), dtype=particles.x.dtype)
        if density:
            self.density_downsampling = density_downsampling
            self.density = np.zeros((self.n_snapshot, particles.n_grid // density_downsampling), dtype=particles.density.dtype)
        if power_spectrum:
            self.logspaced = logspaced
            if n_k_bin:
                self.n_k_bin = n_k_bin
            else:
                ps, _, k_mean, _, _ \
                    = utils.calculate_pseudo_P_k_1d(np.random.rand(particles.n_grid), 
                                                    np.random.rand(particles.n_grid), 
                                                    particles.L*particles.L0,
                                                    logspaced=self.logspaced)
                self.n_k_bin = len(ps)

            self.k_min = k_min
            self.k_max = k_max
            self.power_spectrum = np.zeros((self.n_snapshot, self.n_k_bin), dtype=particles.density.dtype)
            self.power_spectrum_error = np.zeros((self.n_snapshot, self.n_k_bin), dtype=particles.density.dtype)
            self.k_center = np.zeros(self.n_k_bin)
            self.k_mean = np.zeros(self.n_k_bin)
            if linear_powerspectrum:
                self.linear_power_spectrum = np.zeros((self.n_snapshot, self.n_k_bin), dtype=particles.density.dtype)

            #if n_k_bin = None, set self.n_k_bin back to None after creating the storage arrays with the correct size
            if not n_k_bin:
                self.n_k_bin = None

        
    def take_snapshot(self, particles, verbose=False):
        if self.current_snapshot_idx >= self.n_snapshot:
            raise RuntimeError("Too many snapshots. n_snapshot = {}.".format(self.n_snapshot))

        i = self.current_snapshot_idx
        self.a[i] = particles.a
        self.t[i] = particles.t
        self.l[i] = particles.l
        self.beta[i] = particles.beta
        self.Hubble[i] = particles.Hubble(self.a[i])
        self.growth_rate[i] = particles.growth_rate(self.a[i])

        if self.phase_space is not None:
            self.phase_space[i,0] = particles.x[::self.particle_downsampling]
            self.phase_space[i,1] = particles.v[::self.particle_downsampling]
            self.phase_space[i] *= particles.L0
            if particles.time_unit == DT_TIMESTEPS:
                self.phase_space[i,1] *= 1/particles.H0*particles.H0_s*particles.Mpc_m*1.0e-3
            elif particles.time_unit == DLAMBDA_TIMESTEPS:
                self.phase_space[i,1] *= (self.beta[i]*self.l[i]*self.Hubble[i])/particles.H0*particles.H0_s*particles.Mpc_m*1.0e-3
                
        if self.density is not None:
            particles.calculate_density()
            self.density[i] = utils.rebin_1D(particles.density, (particles.n_grid//self.density_downsampling,))

        if self.power_spectrum is not None:
            d = particles.density/np.mean(particles.density) - 1
            ps, ps_error, k_mean, bin_edges, n_mode \
                    = utils.calculate_pseudo_P_k_1d(d, 
                                                    d, 
                                                    particles.L*particles.L0,
                                                    n_k_bin=self.n_k_bin, k_min=self.k_min, k_max=self.k_max, logspaced=self.logspaced)
            self.k_center = np.sqrt(bin_edges[:-1]*bin_edges[1:])
            self.k_mean = k_mean
            self.n_mode = n_mode
            self.power_spectrum[i] = ps
            if self.linear_power_spectrum is not None:
                self.linear_power_spectrum[i] = particles.linear_powerspectrum(self.k_mean*particles.L0)*particles.L0*self.growth_rate[i]**2

        if self.keep_in_memory:
            self.current_snapshot_idx += 1
        else:
            self.save_snapshot(snapshot_idx=0, verbose=verbose)

    def save_snapshot(self, snapshot_idx=0, verbose=False):
        i = snapshot_idx
        if not self.output_directory:
            raise RuntimeError("Output directory not specified.")

        if self.file_format == "npz":
            if self.phase_space is not None:
                filename = self.phase_space_filename_format.format(run_id=self.run_meta_info["id"],
                                                                   L=self.run_meta_info["L_box"],
                                                                   n_particle=self.run_meta_info["n_particle"],
                                                                   a=self.a[i],
                                                                   z=1/self.a[i]-1)
                if verbose: print("Writing phase space snapshot to file {}.".format(filename))
                filename = os.path.join(self.output_directory, filename)
                np.savez(filename, phase_space=self.phase_space[i], a=self.a[i])
            if self.density is not None:
                filename = self.density_filename_format.format(run_id=self.run_meta_info["id"],
                                                               L=self.run_meta_info["L_box"],
                                                               n_particle=self.run_meta_info["n_particle"],
                                                               a=self.a[i],
                                                               z=1/self.a[i]-1)
                if verbose: print("Writing density snapshot to file {}.".format(filename))
                filename = os.path.join(self.output_directory, filename)
                np.savez(filename, density=self.density[i], a=self.a[i], growth_rate=self.growth_rate[i])
            if self.power_spectrum is not None:
                filename = self.power_spectrum_filename_format.format(run_id=self.run_meta_info["id"],
                                                                      L=self.run_meta_info["L_box"],
                                                                      n_particle=self.run_meta_info["n_particle"],
                                                                      a=self.a[i],  
                                                                      z=1/self.a[i]-1)
                if verbose: print("Writing power spectrum snapshot to file {}.".format(filename))
                filename = os.path.join(self.output_directory, filename)
                if not self.linear_power_spectrum is not None:
                    np.savez(filename, power_spectrum=self.power_spectrum[i],
                                       k_mean=self.k_mean,
                                       k_center=self.k_center,
                                       n_mode=self.n_mode,
                                       a=self.a[i])
                else:
                    np.savez(filename, power_spectrum=self.power_spectrum[i],
                                       linear_power_spectrum=self.linear_power_spectrum[i],
                                       k_mean=self.k_mean,
                                       k_center=self.k_center,
                                       n_mode=self.n_mode,
                                       a=self.a[i])
        else:
            raise NotImplementedError("File format {} not supported.".format(self.file_format))




# class Snapshot:
#     def __init__(self, particles,
#                        downsample_particles=1,
#                        downsample_density=1,
#                        powerspectrum=True,
#                        linear_powerspectrum=True,
#                        phases=False,
#                        n_k_bin=None, k_min=None, k_max=None, logspaced=True):
#         self.a = particles.a
#         self.t = particles.t
#         self.l = particles.l
#         self.beta = particles.beta
#         self.Hubble = particles.Hubble(self.a)
#         self.growth_rate = particles.growth_rate(self.a)

#         self.L = particles.L*particles.L0
#         self.n_particle = particles.n_particle

#         self.phasespace = np.vstack((particles.x, particles.v))[:,::downsample_particles]
#         # Convert to dimensionful units again
#         self.phasespace *= particles.L0
#         if particles.time_unit == DT_TIMESTEPS:
#             self.phasespace[1] *= 1/particles.H0*particles.H0_s*particles.Mpc_m*1.0e-3
#         elif particles.time_unit == DLAMBDA_TIMESTEPS:
#             self.phasespace[1] *= (self.beta*self.l*self.Hubble)/particles.H0*particles.H0_s*particles.Mpc_m*1.0e-3

#         particles.calculate_density()
#         self.density = utils.rebin_1D(particles.density, (particles.n_grid//downsample_density,))

#         if phases:
#             dk = np.fft.rfft(particles.density/np.mean(particles.density) - 1)
#             p = np.angle(dk)
#             A = np.abs(dk)
#             self.phases = p
#             self.magnitudes = A

#         if powerspectrum:
#             d = particles.density/np.mean(particles.density) - 1
#             self.powerspectrum, self.powerspectrum_error, self.k_mean, bin_edges, n_mode \
#                     = utils.calculate_pseudo_P_k_1d(d, 
#                                                     d, 
#                                                     self.L,
#                                                     n_k_bin=n_k_bin, k_min=k_min, k_max=k_max, logspaced=logspaced)
#             self.k_center = np.sqrt(bin_edges[:-1]*bin_edges[1:])
#             if linear_powerspectrum:
#                 self.linear_powerspectrum = particles.linear_powerspectrum(self.k_center*particles.L0)*particles.L0*self.growth_rate**2
        


PHASESPACE_PLOT = 0
POWERSPECTRUM_PLOT = 1
DENSITY_PLOT = 2
NO_PLOT = 3

class VisualizeSnapshots:
    def __init__(self, runs, 
                       plots=((PHASESPACE_PLOT, POWERSPECTRUM_PLOT),),
                       plot_size=(4, 3),
                       anim_kwargs={"interval" : 50}):
        self.runs = []

        self.n_y_plots = len(plots)
        self.n_x_plots = len(plots[0])
        self.plot_types = np.array(plots)
        figsize = (self.n_x_plots*plot_size[0], self.n_y_plots*plot_size[1])
        self.fig, self.plots = plt.subplots(self.n_y_plots, self.n_x_plots, figsize=figsize)
        self.plots = np.atleast_2d(self.plots)
        self.fig.subplots_adjust(wspace=0.2, hspace=0.2)


        if type(runs) == list:
            self.runs = runs
        else:
            self.runs = [runs,]

        self.n_snapshots = self.runs[0]["snapshots"].n_snapshot
        self.phasespace_points = []
        self.density_plots = []
        self.powerspectrum_plots = []
        self.linear_powerspectrum_plots = []

        for i, plot in np.ndenumerate(self.plots):
            for j,s in enumerate(self.runs):
                if s["snapshots"].n_snapshot != self.n_snapshots:
                    raise RuntimeError("Unequal number of snapshots in runs!")
                if self.plot_types[i] == PHASESPACE_PLOT:
                    self.phasespace_points.append(self.plots[i].plot([],[], ls="none", marker=",")[0])
                elif self.plot_types[i] == POWERSPECTRUM_PLOT:
                    self.powerspectrum_plots.append(self.plots[i].plot([],[], ls="-")[0])
                    if s["snapshots"].linear_power_spectrum is not None:
                        self.linear_powerspectrum_plots.append(self.plots[i].plot([],[], ls="--")[0])
                elif self.plot_types[i] == DENSITY_PLOT:
                    self.density_plots.append(self.plots[i].plot([],[], ls="-")[0])

        self.animation = matplotlib.animation.FuncAnimation(self.fig, func=self.update_animation, init_func=self.init_animation, 
                               frames=self.n_snapshots, **anim_kwargs)
    
    def init_animation(self):
        for i, plot in np.ndenumerate(self.plots):
            if self.plot_types[i] == PHASESPACE_PLOT:
                plot.grid()
                plot.set_xlim(0, self.runs[0]["snapshots"].run_meta_info["L_box"])
                plot.set_xlabel("x [Mpc h^-1]")
                plot.set_ylabel("v [km s^-1]")
            elif self.plot_types[i] == POWERSPECTRUM_PLOT:
                plot.set_xlim(np.min(self.runs[0]["snapshots"].k_center), np.max(self.runs[0]["snapshots"].k_center))
                plot.set_xscale("log")
                plot.set_yscale("log")
                plot.set_xlabel("k [Mpc^-1 h]")
                plot.set_ylabel("P(k)")
            elif self.plot_types[i] == DENSITY_PLOT:
                plot.set_xlim(0, self.runs[0]["snapshots"].run_meta_info["L_box"])
                plot.set_xlabel("x [Mpc h^-1]")
                plot.set_ylabel("density")
            elif self.plot_types[i] == NO_PLOT:
                plot.axis("off")
    
    def update_animation(self, timestep_idx):
        for i, plot in np.ndenumerate(self.plots):
            if self.plot_types[i] == PHASESPACE_PLOT:
                std_v = 0
                for j, p in enumerate(self.phasespace_points):
                    p.set_data(*self.runs[j]["snapshots"].phase_space[timestep_idx])
                    std_v = max(std_v, np.std(self.runs[j]["snapshots"].phase_space[timestep_idx,1,:]))
                plot.set_ylim(-3*std_v, 3*std_v)

            elif self.plot_types[i] == DENSITY_PLOT:
                max_d = 0
                for j, p in enumerate(self.density_plots):
                    x = np.linspace(0, self.runs[j]["snapshots"].run_meta_info["L_box"], self.runs[j]["snapshots"].density[timestep_idx].size)
                    p.set_data(x, self.runs[j]["snapshots"].density[timestep_idx])
                    max_d = max(max_d, np.max(self.runs[j]["snapshots"].density[timestep_idx]))
                plot.set_ylim(-0.1*max_d, 1.2*max_d)

            elif self.plot_types[i] == POWERSPECTRUM_PLOT:
                max_ps = 0
                for j, p in enumerate(self.powerspectrum_plots):
                    k = self.runs[j]["snapshots"].k_center
                    p.set_data(k, k/pi*self.runs[j]["snapshots"].power_spectrum[timestep_idx])
                    #p.set_data(k, self.runs[i][timestep_idx].powerspectrum, self.runs[i][timestep_idx].powerspectrum_error)
                    max_ps = max(max_ps, np.nanmax(k/pi*self.runs[j]["snapshots"].power_spectrum[timestep_idx]))
                
                for j, p in enumerate(self.linear_powerspectrum_plots):
                    k = self.runs[j]["snapshots"].k_center
                    p.set_data(k, k/pi*self.runs[j]["snapshots"].linear_power_spectrum[timestep_idx])
                    max_ps = max(max_ps, np.nanmax(k/pi*self.runs[j]["snapshots"].linear_power_spectrum[timestep_idx]))
                plot.set_ylim(max_ps*1.0e-3, 3*max_ps)

        self.fig.suptitle("a = {:.2f}, z = {:.1f}".format(self.runs[0]["snapshots"].a[timestep_idx], 1/self.runs[0]["snapshots"].a[timestep_idx]-1))

        return self.phasespace_points, self.density_plots, self.powerspectrum_plots