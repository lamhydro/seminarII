#!/usr/bin/python3
# -*- coding: UTF-8 -*-

__all__ = ['a']
__version__ = '0.1'
__author__ = 'Luis Morales' 


"""
Description
"""

# Import modules
import os,sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import json
from functools import reduce
from sklearn import preprocessing, svm
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# Import other modules

# Function/class definition
def create_mergedf(stinfo,stdir,unit):
  """
  """
  # Read JSON file with station information
  f = open(stinfo)
  stdata = json.load(f) 

  # Read time series
  dfl=[]
  for keys, values in stdata.items():  # Loop throught JSON file dict
    csvfile=os.path.join(stdir, keys+'.csv')
    ts = pd.read_csv(csvfile)
    ts = ts.set_index('Fecha')
    ts.index = pd.to_datetime(ts.index).strftime('%Y-%m-%d')
    #print(ts.columns.tolist()[1])
    if unit in ts.columns.tolist()[0]:
      ts = ts.rename(columns={ts.columns.tolist()[0]: keys})
      dfl.append(ts.iloc[:,0])

  dfm = reduce(lambda  left,right: pd.merge(left,right,how='outer',left_index=True, right_index=True), dfl)
  dfm.index = pd.to_datetime(dfm.index)
  return(dfm)

def plot_ts(dfm,unit):
  """
  """
  #pTSn=pTS[0:4]

  # Plot daily time series complete
  fig = plt.figure()
  ax = plt.subplot(111)
  dfm.plot(ax=ax)
  plt.xlabel('Date')
  if unit=='mm':
    plt.ylabel('Daily precipitation '+'('+unit+')')
  else:
    plt.ylabel('Daily mean streamflow '+'('+unit+')')
  #plt.xticks(rotation=45)
  # Shrink current axis's height by 10% on the bottom
  box = ax.get_position()
  ax.set_position([box.x0, box.y0 + box.height * 0.1,
                   box.width, box.height * 0.9])
  ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.13),
          fancybox=True, ncol=10,fontsize="8", frameon=False)
  plt.grid(color = 'gray', linestyle = '--', linewidth = 0.2)
  plt.show()

  fig = plt.figure()
  ax = plt.subplot(111)
  dfm.boxplot(return_type='axes', ax=ax)
  if unit=='mm':
    plt.ylabel('Daily precipitation '+'('+unit+')')
  else:
    plt.ylabel('Daily mean streamflow '+'('+unit+')')
  plt.grid(color = 'gray', linestyle = '--', linewidth = 0.2)
  plt.xticks(rotation=90)
  plt.show()

  # Plot monthly time series
  if unit=='mm':
    dfm_mo = dfm.resample("M").sum()
  else:
    dfm_mo = dfm.resample("M").mean()
  fig = plt.figure()
  ax = plt.subplot(111)
  dfm_mo.plot(ax=ax)
  plt.xlabel('Date')
  if unit=='mm':
    plt.ylabel('Monthly precipitation '+'('+unit+')')
  else:
    plt.ylabel('Monthly mean streamflow '+'('+unit+')')
  #plt.xticks(rotation=45)
  # Shrink current axis's height by 10% on the bottom
  box = ax.get_position()
  ax.set_position([box.x0, box.y0 + box.height * 0.1,
                   box.width, box.height * 0.9])
  ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.13),
          fancybox=True, ncol=10,fontsize="8", frameon=False)
  plt.grid(color = 'gray', linestyle = '--', linewidth = 0.2)
  plt.show()

  fig = plt.figure()
  ax = plt.subplot(111)
  dfm_mo.boxplot(return_type='axes', ax=ax)
  if unit=='mm':
    plt.ylabel('Monthly precipitation '+'('+unit+')')
  else:
    plt.ylabel('Monthly mean streamflow '+'('+unit+')')
  plt.grid(color = 'gray', linestyle = '--', linewidth = 0.2)
  plt.xticks(rotation=90)
  plt.show()

  # Plot yearly time series
  if unit=='mm':
    dfm_ye = dfm.resample("Y").sum()
  else:
    dfm_ye = dfm.resample("Y").mean()
  fig = plt.figure()
  ax = plt.subplot(111)
  dfm_ye.plot(ax=ax)
  plt.xlabel('Date')
  if unit=='mm':
    plt.ylabel('Yearly precipitation '+'('+unit+')')
  else:
    plt.ylabel('Yearly mean streamflow '+'('+unit+')')
  #plt.xticks(rotation=45)
  # Shrink current axis's height by 10% on the bottom
  box = ax.get_position()
  ax.set_position([box.x0, box.y0 + box.height * 0.1,
                   box.width, box.height * 0.9])
  ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.13),
          fancybox=True, ncol=10,fontsize="8", frameon=False)
  plt.grid(color = 'gray', linestyle = '--', linewidth = 0.2)
  plt.show()

  fig = plt.figure()
  ax = plt.subplot(111)
  dfm_ye.boxplot(return_type='axes', ax=ax)
  if unit=='mm':
    plt.ylabel('Yearly precipitation '+'('+unit+')')
  else:
    plt.ylabel('Yearly mean streamflow '+'('+unit+')')
  plt.grid(color = 'gray', linestyle = '--', linewidth = 0.2)
  plt.xticks(rotation=90)
  plt.show()



def plot_liner(dfm,unit):
  """
  """

  dfm0 = dfm
  coln = dfm.columns.tolist()
  ncoln = len(coln)
  fig = plt.figure()
  gs = fig.add_gridspec(ncoln, ncoln, hspace=0, wspace=0)
  ax=gs.subplots(sharex='col', sharey='row')
  fig.suptitle('Daily precipitation (mm)')
  for n1,i in zip(coln,range(ncoln)):
    for n2,j in zip(coln,range(ncoln)):
      #print(ax[i,j])
      ax[i,j].scatter(dfm[n1],dfm[n2],marker='.')
      ax[i,j].plot(ax[i,j].get_xlim(), ax[i,j].get_ylim(), ls="--", c=".3", linewidth=0.2)
      ax[i,j].set(xlim=(np.nanmin(dfm[n1]), np.nanmax(dfm[n1])), ylim=(np.nanmin(dfm[n2]), np.nanmax(dfm[n2])))

      # Linear regresion
      dfm.fillna(method ='ffill', inplace = True)
      dfm.dropna(inplace = True)
      x = dfm[n1].to_numpy().reshape(-1, 1)
      y = dfm[n2].to_numpy().reshape(-1, 1)
      #print(x)
      model = LinearRegression().fit(x, y)
      r_sq = model.score(x, y)
      print(f"coefficient of determination: {r_sq}")
      if r_sq>=0.5:
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.25)
        y_pred = model.predict(x_test)
        ax[i,j].plot(x_test, y_pred, color ='red')
      #ax[i,j].set_box_aspect(1)
      ax[i,j].tick_params(axis='x', labelsize=6)
      ax[i,j].tick_params(axis='y', labelsize=6)
    ax[i,0].set_ylabel(n1,rotation=0,fontsize=6, labelpad=16)
    ax[ncoln-1,i].set_xlabel(n1, rotation=90,fontsize=6)
  plt.show()
      #dfm.plot(x=i,y=j,style='o')
  #dfm_m1=dfm[dfm.index.month.isin([1])]
  #print(dfm_m1)
 
def plot_simsta(dfm):
  """
  """

  xst = '3503014'
  yst = '3503511'
  dfm0=dfm[[xst,yst]].copy()
  #print(dfm0.isna().sum())


  # Original time series
  fig = plt.figure()
  ax = plt.subplot(111)
  dfm0.plot(ax=ax)
  plt.xlabel('Date')
  plt.show()
  
  fig = plt.figure()
  ax = plt.subplot(111)
  ax.scatter(dfm0[xst],dfm0[yst],marker='.')
  # Linear regresion
  #dfm0.fillna(method ='ffill', inplace = True)
  dfm0.dropna(inplace = True)
  #print(dfm0.isna().sum())
  x = dfm0[xst].to_numpy().reshape(-1, 1)
  y = dfm0[yst].to_numpy().reshape(-1, 1)
  model = LinearRegression().fit(x, y)
  r_sq = model.score(x, y)
  print(f"coefficient of determination: {r_sq}")
  x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.25)
  y_pred = model.predict(x_test)
  ax.plot(x_test, y_pred, color ='red')
  plt.show()

  fig = plt.figure()
  ax = plt.subplot(111)
  #dfm0.plot(kind='hist', density=True,bins=18, alpha=0.5, ax=ax)
  dfm0.plot.density(ax=ax)
  plt.show()

  fig = plt.figure()
  ax = plt.subplot(111)
  dfm0.assign(month=dfm0.index.month).boxplot(by='month', ax=ax)
  #dfm2=dfm0.groupby(['month'],axis=1)
  #dfm2.boxplot()
  #dfm1=dfm.assign(month=dfm.index.month)
  plt.show()

  fig = plt.figure()
  ax = plt.subplot(111)
  dfm0.assign(year=dfm0.index.year).boxplot(by='year', ax=ax)
  #dfm2=dfm0.groupby(['month'],axis=1)
  #dfm2.boxplot()
  #dfm1=dfm.assign(month=dfm.index.month)
  plt.show()




 # MONTHLY time series
  dfm0_mo = dfm0.resample("M").mean()
  fig = plt.figure()
  ax = plt.subplot(111)
  dfm0_mo.plot(ax=ax)
  plt.xlabel('Date')
  plt.show()
  
  fig = plt.figure()
  ax = plt.subplot(111)
  ax.scatter(dfm0_mo[xst],dfm0_mo[yst],marker='.')
  # Linear regresion
  #dfm0.fillna(method ='ffill', inplace = True)
  dfm0_mo.dropna(inplace = True)
  #print(dfm0.isna().sum())
  x = dfm0_mo[xst].to_numpy().reshape(-1, 1)
  y = dfm0_mo[yst].to_numpy().reshape(-1, 1)
  model = LinearRegression().fit(x, y)
  r_sq = model.score(x, y)
  print(f"coefficient of determination: {r_sq}")
  x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.25)
  y_pred = model.predict(x_test)
  ax.plot(x_test, y_pred, color ='red')
  plt.show()

  fig = plt.figure()
  ax = plt.subplot(111)
  #dfm0.plot(kind='hist', density=True,bins=18, alpha=0.5, ax=ax)
  dfm0_mo.plot.density(ax=ax)
  plt.show()

 # MONTHLY time series
  dfm0_ye = dfm0.resample("Y").mean()
  fig = plt.figure()
  ax = plt.subplot(111)
  dfm0_ye.plot(ax=ax)
  plt.xlabel('Date')
  plt.show()
  
  fig = plt.figure()
  ax = plt.subplot(111)
  ax.scatter(dfm0_ye[xst],dfm0_ye[yst],marker='.')
  # Linear regresion
  #dfm0.fillna(method ='ffill', inplace = True)
  dfm0_ye.dropna(inplace = True)
  #print(dfm0.isna().sum())
  x = dfm0_ye[xst].to_numpy().reshape(-1, 1)
  y = dfm0_ye[yst].to_numpy().reshape(-1, 1)
  model = LinearRegression().fit(x, y)
  r_sq = model.score(x, y)
  print(f"coefficient of determination: {r_sq}")
  x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.25)
  y_pred = model.predict(x_test)
  ax.plot(x_test, y_pred, color ='red')
  plt.show()

  fig = plt.figure()
  ax = plt.subplot(111)
  #dfm0.plot(kind='hist', density=True,bins=18, alpha=0.5, ax=ax)
  dfm0_ye.plot.density(ax=ax)
  plt.show()


def plot_aggMandY(dfm,unit):
  """
  """
  #fig = plt.figure()
  #ax = plt.subplot(111)
  #dfm.assign(month=dfm.index.month).boxplot(by='month')
  #dfm1=dfm.assign(month=dfm.index.month)
  #plt.show()

  #dfm1 = dfm.groupby([dfm.index().dtm.month])
  dfm2=dfm1.groupby(['month'],axis=1)
  #dfm1.boxplot()
  print(dfm2)


#class Class1(object):
#    """
#    ...
#    """
#
#    def __init__(self,a=None, b=None):
#
#    def func1(self, data):
#        """
#        ...
#        """

def main():
    """
    Call functions and define classes
    """

    # Create a data frame from multiple precip time series
    dfmp=create_mergedf(jsonstme,stdir,unitp)

    #plot_ts(dfmp,unitp)

    #plot_aggMandY(dfmp,unitp)
    #plot_liner(dfmp,unitp)

    plot_simsta(dfmp)   

if __name__ == "__main__":

    # Define input/output data
    jsonstme='/home/alejandro/Documents/unal/projects/paramoPaper/data/hydro/process/stmetadata.json' # JSON file with station metadata
    stdir='/home/alejandro/Documents/unal/projects/paramoPaper/data/hydro/process' # Directory where station TS are
    unitp='mm'
    unitQ='m3/seg'

    # Call main
    main()
