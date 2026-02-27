import os
import a5client as a5
import pandas as pd
from typing import Optional, Union, List, Tuple

SRC_DIR = os.path.dirname(os.path.abspath(__file__))
CONFIG_FILE_PATH = os.path.join(SRC_DIR,"../config", 'config.json')

#Load Config File (Services and Credentials) --> Credential must be stated in config.py from a5client
def loadConfig(configFile=CONFIG_FILE_PATH):
    service=pd.read_json(configFile)
    return(service)

#Get Serie DF by Id and Type
def getSerie(serieId : int,timeStart,timeEnd,aggStamp : str='D',configFile : str=CONFIG_FILE_PATH ,serieType: str='puntual' ) -> pd.DataFrame:
    service=loadConfig(configFile)
    client=a5.Crud(url=service['api']['url'],token=service['api']['token'])
    serie=client.readSerie(serieId,timeStart,timeEnd,serieType)
    colName=serie['estacion']['nombre']+"_"+serie['var']['var']
    serie=a5.observacionesListToDataFrame(serie["observaciones"])
    serie.index=serie.index.tz_convert(None)
    serie=serie.resample(aggStamp).mean()
    serie.columns=[colName]
    return(serie)

#Get Series DF by Ids and Types
def getSeriesDataFrame(seriesId : List[int],timeStart,timeEnd,aggStamp : str='D' ,configFile: str=CONFIG_FILE_PATH ,seriesTypes: List[str]=['puntual','puntual']) -> pd.DataFrame:
    if len(seriesId) != len (seriesTypes):
        raise ValueError("Series Id and SeriesType must have same length")
    service=loadConfig(configFile)
    client=a5.Crud(url=service['api']['url'],token=service['api']['token'])
    s=[]
    for i in range(0,len(seriesId)):
        v=client.readSerie(seriesId[i],timeStart,timeEnd,seriesTypes[i])
        colName=v['estacion']['nombre']+"_"+v['var']['var']
        v=a5.observacionesListToDataFrame(v["observaciones"])
        v.index=v.index.tz_convert(None)
        v=v.resample(aggStamp).mean()
        v.columns=[colName]
        s.append(v)
    s=pd.concat(s,axis=1)
    return(s)

if __name__ == "__main__":
    import sys
