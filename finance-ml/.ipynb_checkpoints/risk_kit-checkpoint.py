import pandas as pd


def drawdown(series: pd.Series):
    """
    Takes a time series of asset returns
    Computes and returns a data frame that contains:
    1/ wealth index
    2/ previous peaks
    3/ percentage drawdowns
    """
    wealth_index = 1000*(1+series).cumprod()
    previous_peaks = wealth_index.cummax()
    drawdowns = (wealth_index - previous_peaks) / previous_peaks

    return pd.DataFrame({
        "Wealth": wealth_index,
        "Peaks": previous_peaks,
        "Drawdowns": drawdowns
    })


def get_ffm_returns():
    me_m = pd.read_csv("finance-ml-data/Portfolios_Formed_on_ME_monthly_EW.csv", 
                  header=0,
                  index_col=0,
                  parse_dates=True,
                  na_values=-99.99)

    rets = me_m[['Lo 10','Hi 10']]
    rets.columns = ['SmallCap','LargeCap']
    rets = rets/100
    rets.index = pd.to_datetime(rets.index, format='%Y%m').to_period('M')
    return rets



