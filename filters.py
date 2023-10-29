import pandas as pd
import numpy as np  
from pykalman import KalmanFilter
from scipy.signal import butter, sosfilt
from scipy.ndimage import gaussian_filter

class Filter:
    def __init__(self) -> None:
        self.data = pd.DataFrame(pd.read_csv("sample.csv"))

    def mov_average(self) -> pd.DataFrame:
        return self.data.rolling(20).mean()
    
    def expo_smooth(self) -> pd.DataFrame:
        return self.data.ewm(com=0.5).mean()
    
    def kalman(self) -> (pd.DataFrame, pd.DataFrame):
        measurements = np.asarray(enumerate(self.data))
        kf = KalmanFilter(transition_matrices = [[1, 1], [0, 1]], observation_matrices = [[0.1, 0.5], [-0.3, 0.0]])
        kf = kf.em(measurements, n_iter=5)
        (filtered_state_means, filtered_state_covariances) = kf.filter(measurements)
        return pd.DataFrame(filtered_state_means), pd.DataFrame(filtered_state_covariances)
    
    def butterworth(self) -> pd.DataFrame:
        sig = np.asarray(self.data)
        sos = butter(10, 15, 'hp', fs=len(sig), output='sos')
        return pd.DataFrame(sosfilt(sos, sig))

    def gaussian(self) -> pd.DataFrame:
        sig = np.asarray(self.data).reshape(5,5)
        return pd.DataFrame(gaussian_filter(sig, sigma=1))
    
    