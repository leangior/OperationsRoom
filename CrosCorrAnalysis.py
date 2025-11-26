import pandas as pd
import numpy as np
from statsmodels.api import OLS, add_constant
from datetime import datetime,timedelta
from typing import Optional, Union, List, Tuple

def get_cross_cor(up_serie, down_serie, max_lag=10, ini=1)-> pd.DataFrame:
    """Evalúa la correlación cruzada entre una señal de aguas arriba (up_serie) y una señal de aguas abajo (down_serie)
    
    Args:
        up_serie :  
            serie regularizada 'aguas arriba'
        down_serie :  
            serie regularizada 'aguas abajo' 
        ini : 
            retardo inicial
        max_lag: 
            máximo retardo
    Returns:
        Devuelve un dataframe con retardos (índice) y correlaciones (valor)
    """
    r = []
    for i in range(ini, max_lag + 1):
        u_lagged = up_serie.shift(i)
        df = pd.concat([u_lagged, down_serie], axis=1).dropna()
        correlation = df.corr().iloc[1, 0]  # Correlation between lagged up_series and down_serie
        r.append(correlation)
    return pd.DataFrame(r,index=range(ini, max_lag + 1))

def get_response_time(up_serie : Union[pd.Series,pd.DataFrame], down_serie : Union[pd.Series,pd.DataFrame], max_lag : int =10, ini : int =1) -> int:
    """ Evalúa la correlación cruzada entre una señal de aguas arriba (up_serie) y una señal de aguas abajo (down_serie) y devuelve el índice de retardo que maximiza la correlación para la resolución de las señales consideradas
        Args:
        up_serie :  
            serie regularizada 'aguas arriba'
        down_serie :  
            serie regularizada 'aguas abajo' 
        ini : 
            retardo inicial
        max_lag: 
            máximo retardo
        Returns:
            Devuelve el retraso (índice) para el cual se maximiza la correlación cruzada
    """
    r=get_cross_cor(up_serie,down_serie,max_lag,ini).idxmax()
    return(int(r[0]))

def shifted_series_by_best_lag(up_series : Union[pd.Series,pd.DataFrame], down_serie : Union[pd.Series,pd.DataFrame], max_lag : int =10,ini : int =1, freq : str = "infer", verbose : bool = True) -> pd.DataFrame:
        """Evalúa la correlación cruzada entre una señal o conjunto de señales de aguas arriba (up_serie) y una señal de aguas abajo (down_serie), determina el tiempo de traslado más adecuado para la hipótesis de asociación lineal en cada caso y devuelve el dataframe de predictores - desplazados de acuerdo a sus correspondientes retardos -
    Args:
        up_serie :  
            serie o series regularizada 'aguas arriba'
        down_serie :  
            serie regularizada 'aguas abajo' 
        ini : 
            retardo inicial
        max_lag: 
            máximo retardo
        Returns:
            Devuelve un dataframe con las series de aguas arriba desplazadas por su correspondiente lag óptimo 
    """
        X=pd.DataFrame()
        
        if up_series.size == up_series.shape[0]:
            up_series=pd.DataFrame(up_series)
            cols = 1
        else:
            cols = up_series.shape[1]

        for i in range(0,cols):
            best_lag = get_response_time(up_series.iloc[:,i], down_serie, max_lag,ini) 
            if verbose == True :
                print("Serie X"+str(i)+" Best Lag: "+str(best_lag)+" steps")    
            up_serie_shifted = up_series.iloc[:,i].shift(best_lag,freq)
            X = pd.concat([X,up_serie_shifted], axis=1)
            
        X.index=pd.to_datetime(X.index)
        
        return X

def get_lag_and_linear_fit(up_series : Union[pd.Series,pd.DataFrame], down_serie : Union[pd.Series,pd.DataFrame], max_lag : int =10,ini : int =1, verbose : bool = True) -> List:
    """Evalúa la correlación cruzada entre una señal o conjunto de señales de aguas arriba (up_series) y una señal de aguas abajo (down_serie), determina el tiempo de traslado más adecuado para la hipótesis de asociación lineal em cada caso y devuelve el modelo lineal correspondiente (univariante/multivariante) y el dataframe de predictores - desplazados de acuerdo a sus correspondientes retardos óptimos-
    Args:
        up_serie :  
            serie o series regularizada/s 'aguas arriba'
        down_serie :  
            serie regularizada 'aguas abajo' 
        ini : 
            retardo inicial
        max_lag: 
            máximo retardo
        Returns:
            Devuelve una lista con el modelo lineal para el cual se maximiza la correlación cruzada y con el dataframe de predictores 
    """

    X = shifted_series_by_best_lag(up_series,down_serie,max_lag,ini)
    X = pd.concat([X,down_serie],axis=1).dropna()
    y = X.iloc[:,X.shape[1]-1] 
    X = X.iloc[:,0:X.shape[1]-1]
    X = add_constant(X)
    model = OLS(y, X).fit()
    if verbose == True:
         print("Model Adjusted R Square = " + str(model.rsquared))
    return [model,X]

    
def shift_and_adjust(up_series : Union[pd.Series,pd.DataFrame], down_serie : Union[pd.Series,pd.DataFrame], max_lag : int =10,ini : int =1) -> pd.DataFrame:
    """Evalúa la correlación cruzada entre una señal de aguas arriba (up_serie) y una señal de aguas abajo (down_serie), determina el tiempo de traslado más adecuado para la hipótesis de asociación lineal, obtiene el modelo lineal correspondiente y lo ejecuta. Devuelve el vector de valores estiamdos/previstos. 
    Args:
        up_serie :  
            serie regularizada 'aguas arriba'
        down_serie :  
            serie regularizada 'aguas abajo' 
        ini : 
            retardo inicial
        max_lag: 
            máximo retardo
        Returns:
            Devuelve el modelo lineal para el cual se maximiza la correlación cruzada
    """
    model = get_lag_and_linear_fit(up_series, down_serie, max_lag,ini)
    coef = model[0].params.to_numpy()
    sim = model[1].dot(coef)
    return sim

def execute_model(pred_serie : Union[pd.Series,pd.DataFrame], model : Union[pd.DataFrame,np.ndarray] = pd.DataFrame(), obs_serie : Union[pd.Series,pd.DataFrame] = pd.DataFrame(), max_lag : int = 10,ini : int = 1) -> pd.DataFrame:
    """En el caso maś simple se proveen predictores y un modelo (objeto OLS) y lo ejecuta. En este caso no se realiza ningún análisis sobre la base de la serie de predictores (esta se considera como tal, se asume que ya fueron determinados lags y transformaciones óptimas y que el modelo provisto considera esto). En efecto, en este caso simplemente realiza el producto de matrices. En caso que se provea una serie observada determina los lags óptimos y ejecuta el modelo provisto. En caso que se provea una serie observada y no se provea un modelo, realiza el análisis correspondiente, determina lag óptimos, coeficientes de ajuste y ejecuta el modelo obtenido.  
    Args:
        pred_serie :  
            data frame con predictores (vectores columna)
        pars :  
            data frame coeficientes objeto OLS  
        obs_serie : 
            serie observada (opcional)
        Returns:
            Devuelve data frame con valores estimados y observados (opcional)
    """ 

    if hasattr(model,'params'):
         coef = model.params.to_numpy()
    elif model.empty == True:
        if obs_serie.empty == True:
            raise NameError('No model can be found if no obs serie is provided. Check input data')
        else:
            model = get_lag_and_linear_fit(pred_serie,obs_serie,max_lag,ini)
            coef = model[0].params.to_numpy()
         
    if obs_serie.empty == True:
        pred_serie = add_constant(pred_serie)
        sim = pred_serie.dot(coef)
        sim.columns=['sim']
    else :
        pred_serie = shifted_series_by_best_lag(pred_serie,obs_serie,max_lag,ini)
        pred_serie = add_constant(pred_serie) 
        sim = pred_serie.dot(coef)
        sim=pd.concat([sim,obs_serie],axis=1)
        sim.columns=['sim','obs']
    
    return sim

class SimpleMemoryModels():
    """Clase Modelo de Memoria.  Incluye métodos para la calibración y ejecución de modelos - x_{n,j+h}=sum(l=0,L)sum(i=1,n)a_{i,l}x_{i,j-l} -.   
    Args:
        preds :  
            data frame con predictores para sitio de pronosticp
        obs :  
            data frame con observaciones de sitio de pronóstico  
        periodicity : 
            periodicidad de la serie, string ('daily','weekly' or 'monthly')
        calib_period:
            lista con los límites del periodo de calibración
        lead_time:
            int máximo tiempo de antelación
        use_season:
            bool indica si sólo se consideran las tuplas de valores de predictores y observaciones que corresponden al mismo trimestre del año que la fecha de pronóstico
        verbose:
            bool en caso de ser verdadero informa sobre métricas de ajuste al ejecutar los modelos
        k:
            float factor de escala para intervalos de confianza, expresado en sigmas 

    """ 


    def __init__(self,preds : pd.DataFrame, obs : pd.DataFrame, forecast_date : str = 'Now', periodicity : str = 'weekly',calib_period : List[str] = ['2013-07-01','2022-12-31'], lead_time : int = 4, k : float = 1.68, use_season : bool = True, verbose : bool = True):
        
        self.preds = preds
        self.obs = obs
        self.lead_time=lead_time
        self.calib_start = calib_period[0]
        self.calib_end = calib_period[1]
        self.periodicity=periodicity
        self.season = use_season
        self.leadtime = lead_time
        self.k=k
        self.models=[]
        self.verbose=verbose

        if forecast_date == 'Now':
        
            self.forecast_date = str(datetime.now())    

        else:

            self.forecast_date = forecast_date

    def calibrate_models(self):
            
        start=self.calib_start
        end=self.calib_end
        t0=datetime.strptime(self.forecast_date,"%Y-%m-%d")
        forecast_season = (t0.month - 1) // 3 + 1

            

        for horizon in range(0,self.leadtime):

            self.sample = pd.concat([self.preds.loc[start:end],self.obs[start:end].shift(periods=-horizon-1)],axis=1)

            if self.season == True:
            
                self.sample = self.sample[self.sample.index.quarter == forecast_season]

            Y = self.sample.iloc[:,self.sample.shape[1]-1] 
            X = self.sample.iloc[:,0:self.sample.shape[1]-1]
            X = add_constant(X)

            try:

                self.models.append(OLS(Y, X).fit())
                 
                if self.verbose == True:
                    
                    npreds=self.sample.shape[1]-1
                    ndata=self.sample.shape[0]-1
                    
                    print("Memory Model Lag = " + str(horizon+1) + " with " + str(npreds) + " predictors. Data sample size = " + str(ndata))
                    print("Model Adjusted R Square = " + str(self.models[horizon].rsquared))

            except Exception as e:

                print(f"Unexpected error ocurred: {e}")
            
       
            
    def execute_forecast(self):

        self.calibrate_models()
        t0=datetime.strptime(self.forecast_date,"%Y-%m-%d")
        x0=self.obs.loc[self.forecast_date].values[0].tolist()
        predicted_values=[x0]
        confidence_max_values=[x0]
        confindence_min_values=[x0]
        forecasts_dates=[t0]
        

        for horizon in range(0,self.leadtime):
            
            if hasattr(self.models[horizon],'params'):
                
                x = list(self.preds.loc[self.forecast_date])
                x.insert(0,1)
                coef = self.models[horizon].params.to_numpy()
                rmse = np.sqrt(self.models[horizon].scale)

                try: 

                    predicted=np.array(x).dot(coef)
                    predicted_values.append(predicted)
                    confidence_max_values.append(predicted+self.k*rmse)
                    confindence_min_values.append(predicted-self.k*rmse)
                    
                    if self.periodicity == 'weekly': 
                        
                        forecasts_dates.append(t0+timedelta(weeks=horizon+1))
                    
                    elif self.periodicity == 'monthly':

                        forecasts_dates.append(t0+timedelta(months=horizon+1))

                    elif self.periodicity == 'daily':

                        forecasts_dates.append(t0+timedelta(days=horizon+1))
                    
                    else :

                        raise NameError('Peridodicity must be daily, weekly or monthly')
                
                except Exception as e:

                    print(f"Unexpected error ocurred: {e}")

        self.predictions=pd.DataFrame(np.array([confindence_min_values,predicted_values,confidence_max_values]).T,index=forecasts_dates)
        self.predictions.columns=['min','med','max']

if __name__ == "__main__":
    import sys
