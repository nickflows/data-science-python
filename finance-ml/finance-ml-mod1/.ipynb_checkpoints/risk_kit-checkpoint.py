import pandas as pd
import numpy as np
import scipy.stats
from scipy.stats import norm
from scipy.optimize import minimize


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


def get_ffm_d20returns():
    me_m = pd.read_csv("finance-ml-data/Portfolios_Formed_on_ME_monthly_EW.csv", 
                  header=0,
                  index_col=0,
                  parse_dates=True,
                  na_values=-99.99)
    rets = me_m[['Lo 20','Hi 20']]
    rets = rets/100
    rets.index = pd.to_datetime(rets.index, format='%Y%m').to_period('M')
    return rets


def get_hfi_returns():
    """
    load and format the EDHEC Hedge Fund Index Returns
    """
    hfi = pd.read_csv("finance-ml-data/edhec-hedgefundindices.csv", 
                  header=0,
                  index_col=0,
                  parse_dates=True)
    hfi = hfi/100
    hfi.index = hfi.index.to_period('M')
    return hfi 


def skewness(returns):
    """
    Alternative script to scipy.stats.skew()
    Computes the skewness of the supplied series or DataFrame
    Returns a float or a series
    """
    demeaned_returns = returns - returns.mean()
    #use the population standard deviation, so set dof=0
    sigma_returns = returns.std(ddof=0)
    exp = (demeaned_returns**3).mean()
    return exp/sigma_returns**3


def kurtosis(returns):
    """
    Alternative script to scipy.stats.kurtosis()
    Computes the kurtosis of the supplied series or DataFrame
    Returns a float or a series
    """
    demeaned_returns = returns - returns.mean()
    #use the population standard deviation, so set dof=0
    sigma_returns = returns.std(ddof=0)
    exp = (demeaned_returns**4).mean()
    return exp/sigma_returns**4

def is_normal(returns, level=0.01):
    """
    Applies the Jarque-Bera test to determine if a series is normal
    Test is applied at the 1% level by default
    Returns True if the hyptohesis of a normalaity is accepted, False otherwise
    """
    statistic, p_value = scipy.stats.jarque_bera(returns)
    return p_value > level
    


def semideviation(returns):
    """
    Returns the semi-deviation, AKA the negative standard deviation of returns
    returns must be a series or a pandas dataframe
    """
    is_negative = returns < 0
    return r[is_negative].std(ddof=0)


def var_historic(returns, level=5):
    """
    Returns the historic Value at Risk at a specified level
    """
    if isinstance(returns, pd.DataFrame):
        return returns.aggregate(var_historic, level=level)
        
    elif isinstance(returns, pd.Series):
        return -np.percentile(returns, level) 
        
    else:
        raise TypeError("Expected returns to be a series or dataframe")



def var_gaussian(returns, level=5, modified=False):
    """
    Returns the parametric Gaussian VaR of a series or dataframe
    """
    # compute the Z score assuming it was a Gaussian
    z = norm.ppf(level/100)
    if modified:
        s = skewness(returns)
        k = kurtosis(returns)
        z = (z + 
             (z**2 -1)*s/6 +
             (z**3 - 3*z)*(k-3)/24 -
             (2*z**3 - 5*z)*(s**2)/36
            )
    return -(returns.mean() + z*returns.std(ddof=0))

def cvar_historic(returns, level=5):
    if isinstance(returns, pd.Series):
        is_beyond = returns <= -var_historic(returns, level=level)
        return -returns[is_beyond].mean()
    elif isinstance(returns, pd.DataFrame):
        return returns.aggregate(cvar_historic, level=level)
    else:
        raise TypeError("Expected r to be a Series or DataFrame")


#Module 2 (Week 2) Functions

def get_ind_returns():
    """
    Load and format the Ken French 30 Industry Portfolios Value Weighted Monthly Returns
    """
    ind = pd.read_csv("../finance-ml-data/ind30_m_vw_rets.csv", header=0, index_col=0, parse_dates=True)/100
    ind.index = pd.to_datetime(ind.index, format = "%Y%m").to_period('M')
    ind.columns = ind.columns.str.strip()
    return ind


def annualize_rets(r, periods_per_year):
    """
    Annualizes a set of returns
    We should infer the periods per year
    but that is left as an exercise for the class/reader
    """
    compounded_growth = (1+r).prod()
    n_periods = r.shape[0]
    return compounded_growth**(periods_per_year/n_periods)-1

def annualize_vol(r, periods_per_year):
    """
    Annualizes a set of returns
    We should infer the periods per year
    but that is left as an exercise for the class/reader
    """
    return r.std()*(periods_per_year**0.5)

def sharpe_ratio(r, risk_free_rate, periods_per_year):
    """
    Computes the annualized sharpe ratio of a set of returns
    """
    rf_per_period = (1+risk_free_rate)**(1/periods_per_year)-1
    excess_ret = r - rf_per_period
    ann_ex_ret = annualize_rets(excess_ret, periods_per_year)
    ann_vol = annualize_vol(r, periods_per_year)
    return ann_ex_ret / ann_vol


# returns for the portfolio
def portfolio_return(weights, returns):
    """
    Weights to Returns
    """
    return weights.T @ returns


# variance of the portfolio

def portfolio_vol(weights, covmat):
    """
    Weights --> Volatility
    """
    return (weights.T @ covmat @ weights)**0.5


def plot_ef2(npoints, er, cov, style=".-"):
    """
    Plots the 2-asset efficient frontier
    """
    if er.shape[0] != 2 or er.shape[0] != 2:
        raise ValueError("plot_ef2 can only plot 2-asset frontiers")
    weights = [np.array([w,1-w]) for w in np.linspace(0,1,npoints)]
    rets = [portfolio_return(w,er) for w in weights]
    vols = [portfolio_vol(w,cov) for w in weights]
    ef = pd.DataFrame({
        "Returns": rets,
        "Volatility": vols
    })
    return ef.plot.line(x="Volatility", y="Returns", style=style, ylabel="Returns")


def minimize_vol(target_return, er, cov):
    """
    Takes target_return as an input and returns a weight vector that minimizes the
    volatility
    """
    n = er.shape[0]
    init_guess = np.repeat(1/n,n)
    bounds = ((0.0, 1.0),)*n #makes n copies of a tuple or list when you multiply by n
    return_is_target = {
        'type': 'eq',
        'args': (er,),
        'fun': lambda weights, er: target_return - portfolio_return(weights, er)
    }
    weights_sum_to_1 = {
        'type': 'eq',
        'fun': lambda weights: np.sum(weights) - 1
    }
    results = minimize(portfolio_vol,
                       init_guess,
                       args=(cov,),
                       method="SLSQP",
                       options = {'disp': False},
                       constraints = (return_is_target, weights_sum_to_1),
                       bounds = bounds
                      )
    return results.x

def optimal_weights(n_points,er,cov):
    """
    list of weights to run the optimizer on to minimize the volatility
    """
    target_rs = np.linspace(er.min(), er.max(), n_points)
    weights = [minimize_vol(target_return, er, cov) for target_return in target_rs] 
    return weights


def plot_ef(npoints, er, cov, style=".-"):
    """
    Plots the multi-assest efficient frontier
    """
    weights = optimal_weights(npoints, er, cov)
    rets = [portfolio_return(w,er) for w in weights]
    vols = [portfolio_vol(w,cov) for w in weights]
    ef = pd.DataFrame({
        "Returns": rets,
        "Volatility": vols
    })
    return ef.plot.line(x="Volatility", y="Returns", style=style, ylabel="Returns")
    
    
    



    
    


