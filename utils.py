import numpy as np
import matplotlib.animation
import matplotlib.pyplot as plt
import mpl_toolkits.axes_grid1

import scipy.interpolate

pi = np.pi

def rebin_1D(a, shape):
    sh = shape[0],a.shape[0]//shape[0]
    return a.reshape(sh).mean(-1)

def rebin_2D(a, shape):
    sh = shape[0],a.shape[0]//shape[0],shape[1],a.shape[1]//shape[1]
    return a.reshape(sh).mean(-1).mean(1)

def log_bin(array, x, x_min, x_max, n_bin):
    bin_edges = np.logspace(np.log10(x_min), np.log10(x_max), n_bin+1, endpoint=True)
    binned_array = np.zeros(n_bin, dtype=array.dtype)
    mean_x = np.zeros(n_bin, dtype=array.dtype)

    for i in range(n_bin):
        M = np.logical_and(bin_edges[i] <= x, x < bin_edges[i+1])
        binned_array[i] = np.mean(array[M])
        mean_x[i] = np.mean(x[M])

    return binned_array, mean_x

def create_Gaussian_field_1d(P, n_grid, box_size, mean=0, output_FT=False, precision=np.float32):
    # k_min = 2.0*pi/box_size
    # k = np.fft.rfftfreq(n_grid, d=1.0/n_grid)*k_min
    
    # P_grid = P(k)
    # if np.any(P_grid <= 0):
    #     m_ft = np.zeros(k.shape, dtype=np.complex64)
    #     m_ft[P_grid>0] = np.random.normal(scale=np.sqrt((n_grid/box_size)*n_grid*P_grid[P_grid>0]))*np.exp(2j*pi*np.random.random(k.shape)[P_grid>0])
    # else:
    #     m_ft = np.random.normal(scale=np.sqrt((n_grid/box_size)*n_grid*P_grid))*np.exp(2j*pi*np.random.random(k.shape))
    # m_ft[k == 0] = mean
    
    # m = np.fft.irfft(m_ft)
    # return m

    k_grid = np.fft.rfftfreq(n_grid).astype(precision)
    
    k_min = 2*pi/(box_size/n_grid)
    V = box_size/(n_grid)**2
    P_grid = P(k_grid*k_min)
    P_grid[k_grid==0] = mean
    m_ft = np.random.normal(scale=np.sqrt(1/V*P_grid))*np.exp(2j*pi*np.random.random(k_grid.shape))
    m_ft[k_grid == 0] = 0
    m = np.fft.irfft(m_ft)
    if output_FT:
        return m, m_ft, k_grid*k_min
    else:
        return m

def calculate_pseudo_P_k_1d(m1, m2, box_size, n_k_bin=None, k_min=None, k_max=None, logspaced=False):
    if m1.shape != m2.shape:
        raise ValueError("Map dimensions don't match: {}x{} vs {}x{}".format(*(m1.shape + m2.shape)))
        
    m1m2 = np.fft.rfft(m1)*np.conj(np.fft.rfft(m2))
    
    k_grid = np.fft.rfftfreq(m1.shape[0])
    k_min_box = 2*pi/(box_size/m1.shape[0])

    if n_k_bin == None:
        bin_edges = k_grid + k_min_box/2
        Pk_real = m1m2[1:].real
        Pk_imag = m1m2[1:].imag
        Pk_err = np.zeros_like(Pk_real)
        k_mean = k_grid[1:]
        n_mode = np.ones(Pk_real.size, dtype=int)
    else:
        if logspaced:
            bin_edges = np.logspace(np.log10(k_min/k_min_box), np.log10(k_max/k_min_box), n_k_bin+1, endpoint=True)
        else:
            bin_edges = np.linspace(k_min/k_min_box, k_max/k_min_box, n_k_bin+1, endpoint=True)
        n_bin = n_k_bin
    
        Pk_real = np.zeros(n_bin)
        Pk_err = np.zeros(n_bin)
        Pk_imag = np.zeros(n_bin)
        k_mean = np.zeros(n_bin)
        n_mode = np.zeros(n_bin)

        bin_idx = np.searchsorted(k_grid, bin_edges)

        for i in range(n_bin):
            P = m1m2[bin_idx[i]:bin_idx[i+1]]
            Pk_real[i] = np.mean(P.real)
            Pk_imag[i] = np.mean(P.imag)
            Pk_err[i] = np.std(P.real)/len(P)
            k_mean[i] = np.mean(k_grid[bin_idx[i]:bin_idx[i+1]])
            n_mode[i] = len(P)
    
    V = box_size/(m1.shape[0])**2
    return Pk_real*V, Pk_err*V, k_mean*k_min_box, bin_edges*k_min_box, n_mode

def interpolated_powerspectrum_from_file(filename):
    k_grid, P_grid = np.loadtxt(filename, unpack=True)

    log_P_intp = scipy.interpolate.InterpolatedUnivariateSpline(np.log(k_grid), np.log(P_grid), k=1, ext=0)
    def P(k):
        P_k = np.zeros_like(k)
        P_k[k>0] = 1/(2*pi)*k[k>0]**2*np.exp(log_P_intp(np.log(k[k>0])))
        return P_k

    return P

def subplot_colorbar(im, axes, **kwargs):
    cax = mpl_toolkits.axes_grid1.make_axes_locatable(axes).append_axes("right", size = "5%", pad = 0.05)
    plt.colorbar(im, cax=cax, **kwargs)

class AnimatePhaseSpace:
    def __init__(self, snapshots, fig, ax, xlim=None, ylim=None, 
                       trails=False, 
                       formats=[],
                       anim_kwargs={"interval" : 50}):
        self.points = []

        self.ax = ax
        self.xlim = xlim
        self.ylim = ylim

        self.trails = trails

        if type(snapshots) == list:
            self.snapshots = snapshots
        else:
            self.snapshots = [snapshots,]

        self.n_timesteps = self.snapshots[0].shape[0]
        for i,s in enumerate(self.snapshots):
            if s.shape[0] != self.n_timesteps:
                raise RuntimeError("Snapshots with unequal number of timesteps!")
            try:
                f = formats[i]
            except:
                f = {}
            if not "marker" in f:
                f["marker"] = "."
            self.points.append(ax.plot([],[], ls="none", **f)[0])

        self.animation = matplotlib.animation.FuncAnimation(fig, func=self.update_animation, init_func=self.init_animation, 
                               frames=self.n_timesteps, **anim_kwargs)
    
    def init_animation(self):
        if self.xlim:
            self.ax.set_xlim(*self.xlim)
        if self.ylim:
            self.ax.set_ylim(*self.ylim)
        self.ax.grid()
    
    def update_animation(self, timestep_idx):
        std_v = 0
        for i, p in enumerate(self.points):
            if self.trails:
                p.set_data(self.snapshots[i][:timestep_idx,0], self.snapshots[i][:timestep_idx,1])
            else:
                p.set_data(self.snapshots[i][timestep_idx,0], self.snapshots[i][timestep_idx,1])
            std_v = max(std_v, np.std(self.snapshots[i][timestep_idx,1]))
        if not self.ylim:
            self.ax.set_ylim(-3*std_v, 3*std_v)
        return self.points

if __name__ == "__main__":
    print("hello")
    def P(k):
        p = np.zeros_like(k)
        p[k!=0] = k[k!=0]**-1
        return p
    n_grid = 100
    L = 1
    d = create_Gaussian_field_1d(P, n_grid, L)