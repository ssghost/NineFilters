import pandas as pd 

class Filter:
    def __init__(self) -> None:
        self.data = pd.DataFrame(pd.read_csv("sample.csv"))

    def moving_average(self) -> pd.DataFrame:
        return self.data.rolling(20).mean()
    
    def expo_smooth(self) -> pd.DataFrame:
        return self.data.ewm(com=0.5).mean()
    
    def 