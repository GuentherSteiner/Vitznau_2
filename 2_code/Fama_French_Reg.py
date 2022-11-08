import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf


#read data
factorDF = pd.read_csv("Europe_3_Factors.csv")
#drop all rows until 1999
##print(factorDF.iloc[[102]]) #where date = 199901
x = range(0,113)
factorDF.drop(x, inplace=True)
#check that the right data is inplace
##print(factorDF.head())
#reset the index to 0-n
factorDF.reset_index(inplace=True, drop=True)
##print(factorDF.head())


#we now have data from 199912 - 202209 (same as the profits from the momentum strategy)
#next up we regress the excess returns of our momentum strategy against the betas extracted from French's website

#first import the momentum and benchmark returns
mom = pd.read_csv("mom.csv")
benchmark = pd.read_csv("benchmark_return.csv")

#check whether they have the same length, i.e., same number of elements
##print(len(mom) == len(benchmark))
    #outputs True

#create excess returns in a new dataframe
excess_return = {"Date": [], "mom-benchmark": []}
for i in range(0, len(mom)-1):
    excess_return["Date"].append(mom["Dates"][i])
    excess_return["mom-benchmark"].append(mom["Profits"][i]-benchmark["MSCI_EU"][i])
excess_return = pd.DataFrame(excess_return)
#check the first elements, seems as if it has worked
##print(excess_return.head())

#create a new DF with the excess return, and the factors inside
regDF = {"excessReturn": excess_return["mom-benchmark"], "MktRf": factorDF["Mkt-RF"], "SMB": factorDF["SMB"], 
"HML": factorDF["HML"], "RF": factorDF["RF"]}

#now create the linear regression of these excess returns on the beta from the fama french
regression = smf.ols('excessReturn ~ MktRf + SMB + HML + RF', data=regDF).fit()
print(regression.summary())
#the momentum strategy cannot be explained by most of the factors excluding the market factor which can
#is significant

regression2 = smf.ols('excessReturn ~ MktRf', data=regDF).fit()
print(regression2.summary())
#if we look at the AIC we see that we have a better fit if we only take the MktRf to regress on excess return
#log likelihood tiny bit less, but loss function is much higher in above model