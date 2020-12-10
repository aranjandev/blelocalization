'''
BLE receiver, transmitter environment simulation
RSSI modeling reference: 
1. Log shadowing normal model: https://www.cise.ufl.edu/~qathrady/reports/BLE.pdf
2. Using Weibull distribution for noise: https://pdfs.semanticscholar.org/a0b3/67e1ade049f80ff787e1c1b2e9bbc7de6795.pdf
'''

import numpy as np 
from matplotlib import pyplot as plt

SHADOW_NORMAL_RSSI_1 = -40.0
RSSI_UNCERTAINTY_GAUSSIAN_A = 2.0
RSSI_UNCERTAINTY_GAUSSIAN_B = -70
RSSI_UNCERTAINTY_GAUSSIAN_C = 10.0
SHADOW_NORMAL_N = 2
WEIBULL_A = 1

gaussian = lambda x: RSSI_UNCERTAINTY_GAUSSIAN_A * np.exp(-0.5 * ((x-RSSI_UNCERTAINTY_GAUSSIAN_B)/RSSI_UNCERTAINTY_GAUSSIAN_C) * ((x-RSSI_UNCERTAINTY_GAUSSIAN_B)/RSSI_UNCERTAINTY_GAUSSIAN_C))

class Simulator:
    def __init__(self, reader_count):
        self.reader_count = reader_count
        self.all_reader_loc = None 
        self.room_dim = [100,100]

    def add_noise(self, clean):
        uncertainty_coeff = gaussian(clean)
        delta = np.multiply(uncertainty_coeff, np.random.weibull(WEIBULL_A, size=clean.shape))
        return clean + delta

    def init_readers(self, reader_loc=None):
        if self.all_reader_loc is not None:
            self.all_reader_loc = all_reader_loc
        else:
            factors = np.random.rand(self.reader_count,2)
            self.all_reader_loc = self.room_dim * factors

    # convert dist to RSSI
    # dist = [Nx1] array of distances
    # rssi = [Nx1] array of rssi vals calculated as: rssi = -10 * n * log10(dist) + rssi1
    def dist_to_rssi(self, dist):
        rssi_clean = -10 * SHADOW_NORMAL_N * np.log10(dist) + SHADOW_NORMAL_RSSI_1
        rssi = self.add_noise(rssi_clean)
        return rssi
    
    # using log shadow normal
    def rssi_to_dist(self, rssi):
        dist = np.power(10.0, np.subtract(rssi, SHADOW_NORMAL_RSSI_1)/(-10.0 * SHADOW_NORMAL_N))
        return dist

    # estimate RSSI seen at every reader
    def sim_rssi(self, transmitter_loc, n_obs=1, b_draw=False):
        d = np.linalg.norm(self.all_reader_loc-transmitter_loc, axis=1)
        d = np.repeat(d.reshape(1,-1), n_obs, axis=0)
        rssi_vals = self.dist_to_rssi(d)
        if b_draw:
            self.draw(rssi_vals, transmitter_loc[0])
        return rssi_vals

    def draw(self, rssi_vals, transmitter_loc):
        dist = self.rssi_to_dist(rssi_vals)
        fig, ax = plt.subplots()
        plt.xlim(0,200)
        plt.ylim(0,200)
        plt.grid(True)
        ax.set_aspect(1)
        for reader in np.arange(self.reader_count):
            c = self.all_reader_loc[reader,:]
            for obs in np.arange(dist.shape[0]):
                r = dist[obs, reader]
                circ_artist = plt.Circle(c, radius=r, fill=False, linestyle='--', linewidth=0.2, alpha=0.5)
                ax.add_artist(circ_artist)
            plt.scatter(c[0], c[1], marker='o', color='r', linewidths=2.0)
        plt.scatter(transmitter_loc[0], transmitter_loc[1], marker='*', color='b', linewidths=2.0)
        plt.show()

if __name__ == "__main__":
    reader_count = 3
    n_obs = 100
    transmitter_loc = np.asarray([20,20]).reshape(1,2)
    sim = Simulator(reader_count)
    sim.init_readers()
    all_reader_rssis = sim.sim_rssi(transmitter_loc, n_obs, b_draw=True)