import numpy as np
import matplotlib.animation

pi = np.pi

def create_Gaussian_field_1d(P, n_grid, box_size, mean=0):
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

    k_grid = np.fft.rfftfreq(n_grid)
    
    k_min = 2*pi/(box_size/n_grid)
    V = box_size/(n_grid)**2
    
    m_ft = np.random.normal(scale=np.sqrt(1/V*P(k_grid*k_min)))*np.exp(2j*pi*np.random.random(k_grid.shape))
    m_ft[k_grid == 0] = 0
    m = np.fft.irfft(m_ft)
    return m

def calculate_pseudo_P_k_1d(m1, m2, box_size, n_k_bin=None, k_min=None, k_max=None, logspaced=False):
    if m1.shape != m2.shape:
        raise ValueError("Map dimensions don't match: {}x{} vs {}x{}".format(*(map1.shape + map2.shape)))
        
    m1m2 = np.fft.fft(m1)*np.conj(np.fft.fft(m2))
    
    k = np.fft.fftfreq(m1m2.shape[0])
    k_grid = np.abs(k)
    k_min_box = 2*pi/(box_size/m1m2.shape[0])

    if n_k_bin == None:
        bin_edges = np.arange(start=k[1]/1.00001, stop=np.max(k_grid), step=k[1])
        n_bin = len(bin_edges) - 1
    else:
        bin_edges = np.logspace(np.log10(k_min/k_min_box), np.log10(k_max/k_min_box), n_k_bin+1, endpoint=True)
        n_bin = n_k_bin
    
    Pk_real = np.zeros(n_bin)
    Pk_imag = np.zeros(n_bin)
    k_mean = np.zeros(n_bin)
    
    k_sort_idx = np.argsort(k_grid.flatten())
    m1m2_sorted = m1m2[k_sort_idx]
    k_grid_sorted = k_grid[k_sort_idx]
    bin_idx = np.searchsorted(k_grid_sorted, bin_edges)

    for i in range(n_bin):
        P = m1m2_sorted[bin_idx[i]:bin_idx[i+1]]
        Pk_real[i] = np.mean(P.real)
        Pk_imag[i] = np.mean(P.imag)
        k_mean[i] = np.mean(k_grid_sorted[bin_idx[i]:bin_idx[i+1]])
    
    V = box_size/(m1.shape[0])**2
    return Pk_real*V, Pk_imag, k_mean*k_min_box, bin_edges*k_min_box

class AnimatePhaseSpace:
    def __init__(self, snapshots, fig, ax, xlim=(-1, 1), ylim=(-1,1), 
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
            self.points.append(ax.plot([],[], ls="none", marker="o", **f)[0])

        self.animation = matplotlib.animation.FuncAnimation(fig, func=self.update_animation, init_func=self.init_animation, 
                               frames=self.n_timesteps, **anim_kwargs)
    
    def init_animation(self):
        self.ax.set_xlim(*self.xlim)
        self.ax.set_ylim(*self.ylim)
        self.ax.grid()
    
    def update_animation(self, timestep_idx):
        for i, p in enumerate(self.points):
            if self.trails:
                p.set_data(self.snapshots[i][:timestep_idx,0], self.snapshots[i][:timestep_idx,1])
            else:
                p.set_data(self.snapshots[i][timestep_idx,0], self.snapshots[i][timestep_idx,1])
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