import pandas as pd
import numpy as np
import pywt as wt  
from pykalman import KalmanFilter
from scipy.signal import butter, sosfilt, medfilt, savgol_filter
from scipy.ndimage import gaussian_filter
from scipy.fft import fft

class Filter:
    def __init__(self) -> None:
        self.data = pd.DataFrame(pd.read_csv("sample.csv"))

    def mov_average(self) -> pd.DataFrame:
        return self.data.rolling(20).mean()
    
    def expo_smooth(self) -> pd.DataFrame:
        return self.data.ewm(com=0.5).mean()
    
    def kalman(self) -> pd.DataFrame:
        measurements = np.asarray(enumerate(self.data))
        kf = KalmanFilter(transition_matrices = [[1, 1], [0, 1]], observation_matrices = [[0.1, 0.5], [-0.3, 0.0]])
        kf = kf.em(measurements, n_iter=5)
        (filtered_state_means, filtered_state_covariances) = kf.filter(measurements)
        return pd.DataFrame({'fsm':filtered_state_means, 'fsc':filtered_state_covariances})
    
    def butterworth(self) -> pd.DataFrame:
        sig = np.asarray(self.data)
        sos = butter(10, 15, 'hp', fs=len(sig), output='sos')
        return pd.DataFrame(sosfilt(sos, sig))

    def gaussian(self) -> pd.DataFrame:
        sig = np.asarray(self.data).reshape(5,5)
        return pd.DataFrame(gaussian_filter(sig, sigma=1))
    
    def median(self) -> pd.DataFrame:
        return pd.DataFrame(medfilt(np.asarray(self.data), 5))
    
    def wavelet(self) -> pd.DataFrame:
        (ca, cd) = wt.dwt(np.asarray(self.data), 'db1') 
        return pd.DataFrame({'ca':ca, 'cd':cd})
    
    def sav_gol(self) -> pd.DataFrame:
        return pd.DataFrame(savgol_filter(np.asarray(self.data), 5, 2))
    
    def fourier(self) -> pd.DataFrame:
        return pd.DataFrame(fft(np.asarray(self.data))) 