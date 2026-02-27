import numpy as np
import pandas as pd
import statsmodels.api as sm
import datetime
from scipy import stats
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import acf
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.arima.model import ARIMAResults
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.stattools import kpss
from scipy.stats import expon
from scipy.stats import gamma
from scipy.signal import find_peaks
from sklearn.metrics import root_mean_squared_error
from src.CrosCorrAnalysis import get_response_time
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
                raise ValueError('No historical data can be found for the selected period for month '+str(m)+'. Check input data')
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
                raise ValueError('No historical data can be found for the selected period for week '+str(i)+'. Check input data')
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

#hasta aquí todos revisados en la migra y andan OK con series obtenidas medieante GetSeries. LMG 20241122

#--- Módulos para detección de picos, eventos y análisis de tiempos de traslado (requiere CrossCorrAnalysis.py)

def peak_locator(serie : pd.DataFrame, lt_window : int =365, st_window = 30, z_threshold : float =1, value : float = None, centered : bool = True, min_sample_size : int = None, k_size : float = 2/3)-> pd.Series:

    "Given a time series object return peak time series, by standarized anomalies analysis"
    
    if value is None:
        value=serie.min().iloc[0]
    if min_sample_size is None:
        min_sample_size=int(k_size*lt_window)

    u=serie.rolling(window=lt_window,center=centered,min_periods=min_sample_size).mean()
    s=serie.rolling(window=lt_window,center=centered,min_periods=min_sample_size).std()
    z=(serie-u)/s
    z=z.to_numpy().flatten()
    peak_index,_=find_peaks(z,height=z_threshold,distance=st_window)
    peaks=serie.iloc[peak_index,0]

    return(peaks[peaks>=value])

def peak_arrivals_distribution_stats(peak_series : pd.DataFrame,freq : str = 'D',pvalues_thresholds : List[float] = [0.05,0.05])->List[dict]:
    "Compute Peak Arrival Distribution and estimate lambda (mean number of cases per unit time). Also, check stationarity (ADF and KPSS)."
    if freq not in ['h','D']:
        raise ValueError("Freq must be 'D' (from peak time sample obatined by analysis of daily data) or 'h' (hourly)")
    elif freq == 'D':
        factor=365.25
    elif freq =='h':
        factor=365.25*24
        
    
    computed_lambdas=dict()

    time_delta_distribution=peak_series.index.diff()
    time_delta_distribution=(time_delta_distribution.dropna()/np.timedelta64(1,freq))

    p_values=dict()
    p_values['adfuller test']=adfuller(time_delta_distribution)[1]
    p_values['kpss test']=kpss(time_delta_distribution)[1]

    if p_values['adfuller test']<pvalues_thresholds[0]:
        p_values['adf']='Stationary'
    else:
        p_values['adf']='Non Stationary'

    if p_values['kpss test']>pvalues_thresholds[1]:
        p_values['kpss']='Stationary'
    else:
        p_values['kpss']='Non Stationary'



    computed_lambdas['SampleMean']=float(factor/pd.DataFrame(time_delta_distribution).mean().iloc[0])

    exponential_fit_scale=expon.fit(time_delta_distribution,floc=0)[1]
    computed_lambdas['ExpFit']=factor/exponential_fit_scale

    sorted_values=np.sort(time_delta_distribution)
    cumulative_values= np.arange(1,len(sorted_values)+1)/(len(sorted_values)+1)
    empirical_distribution=pd.DataFrame([sorted_values,cumulative_values]).T
    empirical_distribution.columns=['Time','Cumulative Frequency']
    
    X=sm.add_constant(sorted_values)
    Y=np.log(1-cumulative_values)
    m=sm.OLS(Y,X).fit()
    computed_lambdas['LREmpDist']=float(-factor*m.params[1])

    results=[{'Empirical Distribution':empirical_distribution},{'Computed Lambdas':computed_lambdas},{'Stationarity Tests p_values':p_values}]

    return(results)

def computeTr_by_PlottingPosition(x : np.ndarray, a_par : Union[int,float] = None, mean_lambda : Union[int,float]= 1):
    
    v=[]

    if type(x) is not np.ndarray:
        raise TypeError("x must be a 1D array")

    if a_par is None:
            a=0.44  #Grigorten 
    elif type(a_par) is not float:
        if type(a_par) is not int: 
            raise TypeError("Must provide only plotting position unique parameter a (a_par : float/int), for computing original plotting position (i-a)/(N+1-2a). By default is assumed a=0.44 (Grigorten).")
        else:
            a=a_par
    else:
        a=a_par
        
    i=1
    N=len(x)
    x.sort()
    for p in x:
        survival_function_value=1-(i-a)/(N+1-2*a)
        return_period=1/(mean_lambda*survival_function_value)
        v.append([p,return_period])
        i=i+1
        
    v=pd.DataFrame(v)
    v.columns=['peak_value','return_period']
    return(v)

def computeTrDistribution(peaks_serie : Union[pd.Series,pd.DataFrame],a_par : Union[float,int] = None,method : str ='plotting position',pvalues_thresholds : List[float] = [0.05,0.05],confidence_intervals_sample_size : int=100):
    
    v=[]
    
    #computes mean lambda value
    pad=peak_arrivals_distribution_stats(peaks_serie,pvalues_thresholds=pvalues_thresholds)
    mean_lambda=np.array(list(pad[1]['Computed Lambdas'].values())).mean()

    if pad[2]['Stationarity Tests p_values']['adf']=='Stationary' and pad[2]['Stationarity Tests p_values']['kpss']=='Stationary':
        print("Stationarity tests passed (ADF & KPSS)")
    elif pad[2]['Stationarity Tests p_values']['adf'] is not 'Stationary':
        print("ADF tests failed")
    else:
        print("KPSS tests failed")

    if method not in ['plotting position','pareto','extremes']:
        raise ValueError("mehod must be 'plotting position','pareto' or 'extremes'") 
    
    #computes Tr by plotting position of peaks sample
    if method == 'plotting position':
        if a_par is None:
            a=0.44  #Grigorten
        elif type(a_par) is not float:
            if type(a_par) is not int: 
                raise TypeError("Must provide only plotting position unique parameter a (a_par : float/int), for computing original plotting position (i-a)/(N+1-2a). By default is assumed a=0.44 (Grigorten).")
            else:
                a=a_par
        else:
            a=a_par

        i=1
        N=len(peaks_serie)
        for p in peaks_serie.sort_values():
            survival_function_value=1-(i-a)/(N+1-2*a)
            return_period=1/(mean_lambda*survival_function_value)
            v.append([p,return_period])
            i=i+1
        
        v=pd.DataFrame(v)
        v.columns=['peak_value','return_period']
        return(v)
    
    #computes Tr by genpareto or genextreme distribution fitting to peaks sample
    elif method in ['pareto','extremes']:
        if a_par is None:
            if method == 'pareto':
                shape, loc, scale = stats.genpareto.fit(peaks_serie.values)
            elif method == 'extremes':
                shape, loc, scale = stats.genextreme.fit(peaks_serie.values)
        else:
            if type(a_par) is not float or int:
                raise Warning("Avoiding use of a_par (not neccesary with 'pareto' or 'extremes' option - distribution parameters are oibtained by fitting-). NEsidea a_par must be a single float type variable")
            else:
                raise Warning("Avoiding use of a_par (not neccesary with 'pareto' or 'extremes' option - distribution parameters are oibtained by fitting-)")
    
        for p in peaks_serie.sort_values():
            if method == 'pareto':
                return_period=1/(mean_lambda*stats.genpareto.sf(p,shape,loc,scale))
            elif method == 'extremes':
                return_period=1/(mean_lambda*stats.genextreme.sf(p,shape,loc,scale))
            v.append([p,return_period])

        v=pd.DataFrame(v)
        v.columns=['peak_value','return_period']
        return(v)



def computeTr(x : float, serie : pd.DataFrame,freq : str = 'D', lt_window : int =365, st_window = 30, z_threshold : float =1, value : float = None, centered : bool = True, min_sample_size : int = None, k_size : float = 2/3,pvalues_thresholds=[0.05,0.05])->dict:


    peaks = peak_locator(serie,lt_window,st_window,z_threshold,value,centered,min_sample_size,k_size)
    etas = peak_arrivals_distribution_stats(peaks,freq,pvalues_thresholds)
    
    pars=dict()

    mean_lambda = np.array(list(etas[1]['Computed Lambdas'].values())).mean()
    pars['mean lambda']=mean_lambda
    pars['adf']=etas[2]['Stationarity Tests p_values']['adf']
    pars['kpss']=etas[2]['Stationarity Tests p_values']['kpss']

    tr=dict()

    shape, loc, scale = stats.genpareto.fit(peaks.values)
    tr['genpareto return period'] = 1/(mean_lambda*stats.genpareto.sf(x,shape,loc,scale))
    pars['genpareto scipy.pars']=shape,loc,scale

    shape, loc, scale = stats.genextreme.fit(peaks.values)
    tr['genxtremes return period'] = 1/(mean_lambda*stats.genextreme.sf(x,shape,loc,scale)) 
    pars['genextreme scipy.pars']=shape,loc,scale

    results=[{'value':x},{'computed Tr':tr},{'computed pars':pars},{'peaks serie':peaks}]

    return(results)

#Continuar, falta agregarle distribuciones, quizàs pensar en Bootstraping en pasos anteriores (estimacón robusta lambda) o inclusive aquì (estimación de Trs con bootstraping en fitting). Ver además si no conviene siempre devolver la empírica para contraste y tambièn el resultado de los test de estacionariedad para evaluar calidad de estimación

def hydrograph_locator(serie : pd.DataFrame, lt_window : int =365, st_window = 30, z_threshold : float =1, value : float = None, centered : bool = True, min_sample_size : int = None, k_size : float = 2/3)-> dict:

    "Given a time series object return a collection of event hydrographs, obtained by peak flow standarized anomalies analysis"

    hydrographs=[]
    tp=[]
    frequency=serie.index.diff().value_counts().idxmax()

    peaks=peak_locator(serie=serie,lt_window=lt_window,st_window=st_window,z_threshold=z_threshold,value=value,centered=centered,min_sample_size=min_sample_size,k_size = k_size)


    for i in range(0,peaks.shape[0]):
        #timedelta on days units and st_window on timeseries timestep units (peak_locator), hence frequency it is used to scale factors
        t0 = peaks.index[i] - frequency*st_window/2 
        t1 = peaks.index[i] + frequency*st_window/2
        tp.append(peaks.index[i])
        hydrographs.append(serie.loc[t0:t1])
        
    tp=pd.Series(tp)

    return(dict(Tp=tp,hydrographs=hydrographs))

def hydrograph_stats(hydrograph,tp : int = None)-> dict:

    "Computes prominence of event hydrograph"

    t0 = hydrograph.index.min()
    t1 = hydrograph.index.max()

    if tp is None:
        tp = hydrograph.idxmax().iloc[0]
    
    u_1 = hydrograph.loc[t0:tp].min().iloc[0]
    u_2 = hydrograph.loc[tp:t1].min().iloc[0]
    base_line=max(u_1,u_2)

    prominence=hydrograph.max()-base_line
    mean_value=hydrograph.mean()
    ratio=prominence/mean_value

    return(dict(prominence=prominence.iloc[0],mean_value=mean_value.iloc[0],ratio=ratio.iloc[0]))

def hydrograph_size_fitting(hydrograph : pd.DataFrame, tp : float):
    
    t0 = hydrograph.index.min()
    t1 = hydrograph.index.max()
    tx = hydrograph.idxmax().iloc[0]

    while tx != tp: 
        
        if tx > tp:
            t = hydrograph.loc[tp:tx].idxmin().iloc[0]
        else:
            t = hydrograph.loc[tx:tp].idxmin().iloc[0]
        
        if (hydrograph.loc[t]-hydrograph.loc[tp]).iloc[0] == 0: #It is a plateau, hence tp=tx (to avoid infinite looping)
            if t > tp:
                t1=hydrograph.loc[t:t1].idxmin().iloc[0]
            else:
                t0=hydrograph.loc[t0:t].idxmin().iloc[0]                
            hydrograph = hydrograph.loc[t0:t1]
            tx=tp
        else:
            if t > tp:
                t1 = t
            else:
                t0 = t
            hydrograph = hydrograph.loc[t0:t1]
            tx = hydrograph.idxmax().iloc[0]

    return(hydrograph)



def tau_computation(upstream_hydrographs: List[pd.DataFrame], downstream_serie : pd.DataFrame, peak_times : pd.Series = None, max_lag : int = None,ini : int=0, k : float = 1/2)-> pd.DataFrame:


    v=[]
    tp = None

    if k > 1:
        raise ValueError("k must be positive equal or less than 1")
    
    if len(upstream_hydrographs) != len(peak_times):
        raise ValueError("Peak times sample size must be the same than hydrographs sample size")

    for i in range(0,len(upstream_hydrographs)):
        
        if peak_times is None:
            tp=upstream_hydrographs[i].idxmax().iloc[0]
        else:
            tp=peak_times.iloc[i]
        
        if max_lag is None:
            max_lag=int(k*upstream_hydrographs[i].shape[0])

        upstream_hydrographs[i]=hydrograph_size_fitting(hydrograph=upstream_hydrographs[i],tp=tp)    
        stats=hydrograph_stats(hydrograph=upstream_hydrographs[i],tp=tp)
        indexes=upstream_hydrographs[i].index
        
        tau=get_response_time(up_serie=upstream_hydrographs[i].loc[indexes],down_serie=downstream_serie.loc[indexes],max_lag=max_lag,ini=ini)
        prominence=stats['prominence']
        mean_value=stats['mean_value']
        ratio=stats['ratio']
        v.append([tp,tau,mean_value,prominence,ratio])
    
    v=pd.DataFrame(v,columns=['event_date','tau','mean_value','prominence','ratio'])

    return(v)


if __name__ == "__main__":
    import sys

