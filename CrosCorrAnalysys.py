import pandas as pd
import numpy as np
from statsmodels.api import OLS, add_constant
from typing import Optional, Union, List, Tuple

def get_cross_cor(up_series, down_series, max_lag=10, ini=1)-> pd.DataFrame:
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
        u_lagged = up_series.shift(i)
        df = pd.concat([u_lagged, down_series], axis=1).dropna()
        correlation = df.corr().iloc[1, 0]  # Correlation between lagged up_series and down_series
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

def get_lag_and_linear_fit(up_series : Union[pd.Series,pd.DataFrame], down_serie : Union[pd.Series,pd.DataFrame], max_lag : int =10,ini : int =1, verbose : bool = True) -> List:
    """Evalúa la correlación cruzada entre una señal de aguas arriba (up_serie) y una señal de aguas abajo (down_serie), determina el tiempo de traslado màs adecuado para la hipótesis de asociación lineal y devuelve el modelo lineal correspondiente y el dataframe de predictores
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
            Devuelve una lista con el modelo lineal para el cual se maximiza la correlación cruzada y con el dataframe de predictores 
    """
    X=pd.DataFrame()

    for i in range(0,up_series.shape[1]):
        best_lag = get_response_time(up_series.iloc[:,i], down_serie, max_lag,ini) 
        if verbose == True :
            print("Serie X"+str(i)+" Best Lag: "+str(best_lag)+" steps")    
        up_serie_shifted = up_series.iloc[:,i].shift(best_lag)
        X = pd.concat([X,up_serie_shifted], axis=1)

    X = pd.concat([X,down_serie],axis=1).dropna()
    y = X.iloc[:,X.shape[1]-1] 
    X = X.iloc[:,0:X.shape[1]-1]
    X = add_constant(X)
    model = OLS(y, X).fit()
    return [model,X]
    
def shift_and_adjust(up_series : Union[pd.Series,pd.DataFrame], down_serie : Union[pd.Series,pd.DataFrame], max_lag : int =10,ini : int =1) -> pd.DataFrame:
    """Evalúa la correlación cruzada entre una señal de aguas arriba (up_serie) y una señal de aguas abajo (down_serie), determina el tiempo de traslado màs adecuado para la hipótesis de asociación lineal, obtiene el modelo lineal correspondiente y lo ejecuta
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

if __name__ == "__main__":
    import sys
