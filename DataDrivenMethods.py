import numpy as np
import pandas as pd
import statsmodels.api as sm
import datetime
from scipy import stats
from statsmodels.tsa.seasonal import seasonal_decompose
from scipy.stats import gamma
from sklearn.metrics import root_mean_squared_error
from statsmodels.tsa.stattools import acf
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.arima.model import ARIMAResults
from typing import Optional, Union, List, Tuple

#Library of Data Driven Methods for S2S product generation
 
def getFitScores(historical : pd.DataFrame, X : float, use_logs :bool =False, distribution_type: str='theoretical') -> pd.DataFrame:
   
    """Statistical computing of anomaly scores from X (serie) by using historical data (at predefined aggregation level). Anomalies are expressed as normalized deviations scores above/bellow the mean value"""
   
    if distribution_type == 'empiricalUK':
        use_logs = True
    
    if use_logs == True:
        historical = np.log(historical)
        X = np.log(X)
    
    if distribution_type == 'theoretical':
        # Fit the gamma distribution to the historical data
        shape, loc, scale = stats.gamma.fit(historical, floc=0)  # fit with fixed location parameter
        # Calculate the scores using the quantile function
        scores = stats.norm.ppf(stats.gamma.cdf(X, a=shape, scale=scale))
    
    elif distribution_type == 'empiricalUK':
        u = np.mean(historical)
        s = np.std(historical,axis=0)
        scores = (X - u) / s
    
    # Create a pandas Series to display the scores with the index of X
    # scoresSeries = pd.Series(scores, index=X.index)
    scoresSeries = pd.DataFrame(data=scores, index=X.index)
    return scoresSeries.sort_index().dropna()


def getValue(serie : pd.DataFrame, scores : pd.DataFrame, method : str ='monthly', use_logs : bool=False, distribution_type : str ='theoretical', fN : int =30, start : str ='init') -> pd.DataFrame:

    """Statistical computing of signal values from anomaly scores by using historical data (aggregation level must be declared on method, i.e monthly, weekly)"""

    X = []

    # Set the start year (reference period)
    if start == 'init':
        st = serie.index.min().year
        st = datetime.datetime.strptime(str(st),'%Y')
    else:
        st = datetime.datetime.strptime(str(start),'%Y')
    
    # Set the end year (reference period)
    ed = st +  pd.DateOffset(years=fN)
    
    for t in scores.index:
        if method == 'monthly':
            m = t.month
            sub = serie[serie.index.month == m].loc[st:ed]
            sub = sub.dropna()
            if len(sub) == 0:
                raise NameError('No historical data can be found for the selected period for month '+str(m)+'. Check input data')
        elif method == 'weekly':
            w = t.isocalendar()[1]  # Get the week number of the year
            sub = serie[serie.index.isocalendar().week == w].loc[st:ed]
            sub = sub.dropna()
            if len(sub) == 0:
                raise NameError('No historical data can be found for the selected period for week '+str(w)+'. Check input data')
        
        if distribution_type == 'theoretical':
            # Fit the gamma distribution to the historical series
            shape, loc, scale = stats.gamma.fit(sub, floc=0)
            # Calculate the theoretical value
            norm_cdf_value = stats.norm.cdf(scores[t])
            x = stats.gamma.ppf(norm_cdf_value, a=shape, scale=scale)
        elif distribution_type == 'empiricalUK':
            u = np.mean(sub)
            s = np.std(sub,axis=0)
            x = u + scores[t] * s
        
        if use_logs == 'T':
            x = np.exp(x)
        
        X.append(x)
    
    # Creating a pandas Series with the calculated values
    Xseries = pd.DataFrame(data=X, index=scores.index)
    return Xseries.sort_index().dropna()


def getCivilAnom(xts : pd.DataFrame, fN : int =30, method : str ='weekly', use_logs : bool =False, anom_type : str ='empiricalUK', start : str ='init') -> pd.DataFrame:

    """Computes anomaly scores series (fixed temporal windows based on 'civil  calendary')"""
    
    v = []

    # Set the start year (reference period)
    if start == 'init':
        st = xts.index.year.min()
        st = datetime.datetime.strptime(str(st),'%Y')
    else:
        st = datetime.datetime.strptime(str(start),'%Y')
    
    # Set the end year (reference period)
    ed = st +  pd.DateOffset(years=fN) 
    
    if method == 'monthly':
        m = xts.resample('M').mean()
        nMonth = m.index.month.unique().values
        for i in nMonth:
            historical = m[m.index.month == i].loc[st:ed]
            historical=historical.dropna()
            if len(historical) == 0:
                raise NameError('No historical data can be found for the selected period for month '+str(m)+'. Check input data')
            X = m[m.index.month == i]
            scores = getFitScores(historical, X, use_logs, anom_type)
            v.append(scores)
    elif method == 'weekly':
        s = xts.resample('W').mean()
        nWeek = s.index.isocalendar().week.unique()
        for i in nWeek:
            historical = s[s.index.isocalendar().week == i].loc[st:ed]
            historical=historical.dropna()
            if len(historical) == 0:
                raise NameError('No historical data can be found for the selected period for week '+str(i)+'. Check input data')
            X = s[s.index.isocalendar().week == i]
            scores = getFitScores(historical, X, use_logs, anom_type)
            v.append(scores)

    anomSerie = pd.concat(v)
    return anomSerie.sort_index()


def getAnalogiesScores(anom_serie : pd.DataFrame, forecast_date : str =None, back_step : int =6, for_step : int =13, M=6, null_value : int =9999) -> dict:
    
    """Return analogies anomaly scores series and analysis results from anomaly serie """
    
    if forecast_date is None:
        forecast_date = anom_serie.index.max()
    else:
        forecast_date=datetime.datetime.strptime(str(forecast_date),'%Y-%m-%d')

    freq = round(anom_serie.index.to_series().diff().mean().days)
    
    # Set the frequency of the series 
    if 6 <= freq <= 8:
        method = 'weekly'
    elif 29 <= freq <= 31:
        method = 'monthly'
    else:
        raise ValueError("Anomalies must be weekly or monthly")
    
    if method == 'monthly':
        interval = pd.date_range(forecast_date - pd.DateOffset(months=back_step), forecast_date, freq='M')
        t = pd.date_range(forecast_date - pd.DateOffset(months=back_step), 
                          forecast_date + pd.DateOffset(months=for_step) - pd.DateOffset(days=1), freq='M')
    elif method == 'weekly':
        interval = pd.date_range(forecast_date - pd.DateOffset(weeks=back_step), forecast_date, freq='W')
        t = pd.date_range(forecast_date - pd.DateOffset(weeks=back_step), 
                          forecast_date + pd.DateOffset(weeks=for_step) - pd.DateOffset(days=1), freq='W')

    st=interval.min()
    ed=interval.max()
    obs = anom_serie.loc[st:ed][:back_step]
    errors = []
    valid_periods = []
    results = {}

    for y in anom_serie.index.year.unique()[anom_serie.index.year.unique()<forecast_date.year]:
        i=forecast_date.year-y
        if method == 'monthly':
            interval_i = pd.date_range(forecast_date - pd.DateOffset(years=i, months=back_step), 
                                        forecast_date - pd.DateOffset(years=i), freq='M')
            valid_period_i = pd.date_range(forecast_date - pd.DateOffset(years=i, months=back_step), 
                                            forecast_date - pd.DateOffset(years=i) + pd.DateOffset(months=for_step), freq='M')
        elif method == 'weekly':
            interval_i = pd.date_range(forecast_date - pd.DateOffset(years=i, weeks=back_step), 
                                        forecast_date - pd.DateOffset(years=i), freq='W')
            valid_period_i = pd.date_range(forecast_date - pd.DateOffset(years=i, weeks=back_step), 
                                            forecast_date - pd.DateOffset(years=i) + pd.DateOffset(weeks=for_step), freq='W')

        st=interval_i.min()
        ed=interval_i.max()
        if len(obs) == len(anom_serie.loc[st:ed]):
            error = root_mean_squared_error(obs, anom_serie.loc[st:ed])
        else:
            error = null_value
            
        errors.append(error)
        valid_periods.append(valid_period_i)

    sorted_errors = np.argsort(errors)
    valid_periods = [valid_periods[i] for i in sorted_errors[:M]]
     
    results['metrics'] = np.array(errors)[sorted_errors[:M]]
    results['analogies'] = np.vstack([anom_serie.loc[p].squeeze() for p in valid_periods]) 
    results['analogies_central_trend'] = pd.DataFrame(results['analogies'].mean(axis=0),index=t)
    results['analogies'] = pd.DataFrame(np.transpose(results['analogies']),index=t)
    results['obs'] = obs
    results['validPeriods']=[]
    for item in valid_periods:
        results['validPeriods'].append([min(item),max(item)])
    
    return results


def getAnalogiesValues(analogies : dict, obs : pd.DataFrame) -> pd.DataFrame:
    
    """Compute original signal values from analogies scores values"""
    
    dims=analogies['analogies'].shape
    analogies_values = np.empty(shape=dims)

    freq = round(analogies['analogies'].index.to_series().diff().mean().days)
    
    if 6 <= freq <= 8:
        obs = obs.resample('W').mean()
    elif 29 <= freq <= 31:
        obs = obs.resample('M').mean()
    
    j=0
    for period in analogies['validPeriods']:
        st=period[0]
        ed=period[1]
        analogies_values[:,j]=obs.loc[st:ed].values[:,0]
        j=j+1
    
    analogies_values_df = pd.DataFrame(analogies_values, index=analogies['analogies'].index)
    
    return analogies_values_df

 
def getCentralTrendandForecasts(analogies_forecast_df : pd.DataFrame,obs : pd.DataFrame,k: float=2,maxSampleSize: int=6)-> dict:
    
    """Compute analoguies central trend and get bias adjusted forecasts (firstly by using analogies signal forecasts, obtained from getAnalogiesValues, and applying inverse distance weighting - if uniform weighting is desired k must be set on 0-, secondly by using linear fit parameters on the original analogies values)"""

    freq = round(analogies_forecast_df.index.to_series().diff().mean().days)
    
    w=[]
    results={}
    st=analogies_forecast_df.index.min()

    if 6 <= freq <= 8:
        obs = obs.resample('W').mean()
        ed=st+pd.DateOffset(weeks=maxSampleSize)
    elif 29 <= freq <= 31:
        obs = obs.resample('M').mean()
        ed=st+pd.DateOffset(months=maxSampleSize)

    sampleObs=obs.loc[st:ed]

    for col in analogies_forecast_df:
        w.append(root_mean_squared_error(analogies_forecast_df[col].iloc[0:len(sampleObs)],sampleObs))
    
    w=np.array(w)
    w=(1/w**k)/sum(1/w**k)
    w_central_trend=np.dot(np.array(analogies_forecast_df),w)
    w_central_trend=pd.DataFrame(w_central_trend,index=analogies_forecast_df.index)
    
    X=np.array(w_central_trend[0:len(sampleObs)])
    Y=np.array(sampleObs)
    X=sm.add_constant(X)
    m=sm.OLS(Y,X).fit()

    ed=analogies_forecast_df.index.max()
    obs=obs.loc[st:ed]

    results['weights']=w
    results['centralTrend']=w_central_trend
    results['centralTrendBiasAdjusted']=m.params[0]+m.params[1]*w_central_trend
    results['rmse']=(m.mse_resid)**0.5
    results['analogies']=analogies_forecast_df
    results['forecastsAdj']=m.params[0]+m.params[1]*analogies_forecast_df
    results['stdForecasts']=results['forecastsAdj'].std(axis=1)
    results['obsSerie']=obs
    results['linear_model_pars']=m.params
    results['linear_model_rsquared']=m.rsquared

    return(results)


def persistenseCorrGram(anom_serie : pd.DataFrame, max_lag : int =6) -> pd.DataFrame:
    
    """Computes anomalies cross-correlogram and linear fit statistics for persistence forecasts (test persistence)"""
    
    lags = []
    r2 = []
    offset = []
    bias = []
    
    for i in range(1, max_lag + 1):
        pred = anom_serie[:-i].values
        lagged_obs = anom_serie[i:].values
        
        pred = sm.add_constant(pred)  # Add intercept vector c=(1,1...,1)
        model = sm.OLS(lagged_obs,pred).fit()
        
        lags.append(i)
        r2.append(model.rsquared)
        offset.append(model.params[0])  # Intercept
        bias.append(model.params[1])    # Slope
    
    return pd.DataFrame({'lag': lags, 'r2': r2, 'offset': offset, 'bias': bias})

def getPersistenceForecast(serie, timestart, score, use_logs=False, forecast_type='empiricalUK', method='weekly', fN=30, maxlag=1, start='init') -> pd.DataFrame:
    
    """Compute Anomaly Persistence Forecast"""

    timestart = pd.to_datetime(timestart)
    X = []
    t = []
    
    # Monthly or weekly grouping
    if method == 'monthly':
        historical = serie.resample('M').mean()
    elif method == 'weekly':
        historical = serie.resample('W').mean()
    
    # Set the range of historical data
    if start == 'init':
        st = historical.index.min().year
        st = datetime.datetime.strptime(str(st),'%Y')
    else:
        st = datetime.datetime.strptime(str(start),'%Y')
    ed = st +  pd.DateOffset(years=fN) 
    
    if forecast_type == 'empiricalUK':
        use_logs = True
    
    if use_logs:
        historical = np.log(historical)
    
    for i in range(1, maxlag + 1):
        if method == 'monthly':
            t_point = timestart + pd.DateOffset(months=i)
            imonth = t_point.month 
            subset = historical[historical.index.month == imonth].loc[st:ed]
        elif method == 'weekly':
            t_point = timestart + pd.DateOffset(weeks=i)
            iweek = t_point.isocalendar()[1]
            subset = historical[historical.index.isocalendar().week == iweek].loc[st:ed]
        
        if forecast_type == 'theoretical':
            # Gamma fit
            shape, loc, scale = gamma.fit(subset.dropna(),floc=0)
            X.append(gamma.ppf(stats.norm.cdf(score), shape, loc=loc, scale=scale))
        elif forecast_type == 'empiricalUK':
            # Empirical Fit
            mean_val = subset.mean()
            std_val = subset.std()
            X.append(mean_val + score * std_val)
        
        if use_logs:
            X[-1] = np.exp(X[-1])
        
        t.append(t_point)
    
    forecast_series = pd.DataFrame(X, index=pd.to_datetime(t))
    return forecast_series

#hasta aquí todos revisados en la migra y andan OK con series obtenidas medieante GetSerires. LMG 20241122


def get_auto_arima_forecast(anom, method='monthly'):

    """Anomaly signal ARIMA model computation and forecasting generation (by autoarima). Indeed if persistence holds, autocorrelation must be significative, so... let's test an ARIMA!"""
    
    model = ARIMA(anom, order=(5,1,0))  # auto_arima functionality
    model_fit = model.fit()
    forecast = model_fit.get_forecast(steps=12)
    central = forecast.predicted_mean
    conf_int = forecast.conf_int()
    
    t0 = anom.index.max()
    horizon = len(central)
    
    if method == 'monthly':
        t = pd.date_range(t0 + pd.DateOffset(months=1), periods=horizon, freq='M')
    elif method == 'weekly':
        t = pd.date_range(t0 + pd.DateOffset(weeks=1), periods=horizon, freq='W')
    
    forecast_df = pd.DataFrame({
        'lo2': conf_int.iloc[:, 0],
        'lo1': conf_int.iloc[:, 1],
        'central': central,
        'up1': conf_int.iloc[:, 2],
        'up2': conf_int.iloc[:, 3]
    }, index=t)
    
    return forecast_df

if __name__ == "__main__":
    import sys

