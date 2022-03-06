import statsmodels.api as sm

import pandas as pd
import numpy as np

import Scripts.GAFunctions as GAfunctions


def dispersion_calc(mod, y, X):
    
    expected = mod.predict(X)
    E2 = sum((y-expected)**2/expected)
    dispersion = E2/(len(y) - len(mod.params))
    
    return dispersion


def stat_result(mod, X, y):
    ### AIC
    output_AIC = mod.aic

    ### Dispersion
    output_DIS = dispersion_calc(mod, y, X)
    
#     print("Dispersion: ", dispersion_calc(res_LR, y, X_with_constant))

    ### R-squared
    sst = sum(map(lambda X: np.power(X,2),y-np.mean(y))) 
    sse = sum(map(lambda X: np.power(X,2),mod.resid)) 
    r2 = 1.0 - sse/sst
#     print("R-squared: ", r2)

    output_RSQ = r2
    
    return output_AIC, output_DIS, output_RSQ




def fitness(df_x, n_clusters):
    
    df_stat, SS = GAfunctions.fitness(df_x, n_clusters)
    
    Clusters_count = list(df_stat.Clusters.value_counts())

    df_stat = pd.get_dummies(df_stat, columns=['Clusters'])
    
    variables_col = list(df_stat.columns[9:])
    variables_col.append('num_R')
    variables_col.append('num_T')
    
#     X = df_stat[['Clusters_2','Clusters_3']]
    y = (df_stat['Extra_Steps'].astype(float)).to_numpy()
    
#     X_with_constant = sm.add_constant(X)
#     res_LR = sm.OLS(y, X_with_constant).fit()
    
#     df_stat['Residual'] = res_LR.resid
    
#     X = df_stat[['Clusters_2','Clusters_3','Residual']]
    X = df_stat[variables_col]
    X_with_constant = sm.add_constant(X)
    
    mod_glm = sm.ZeroInflatedPoisson(endog=y, exog=X_with_constant, exog_infl=X_with_constant.iloc[:,0:3], 
                                   inflation='logit')
    res_glm = mod_glm.fit_regularized(disp=False)
#     print(res_glm_3.summary())
    
    
    AIC, DIS, RSQ = stat_result(res_glm, X_with_constant, y)
    
#     print("-----------------------")
#     print("AIC: ", round(AIC, 2))
#     print("DIS: ", round(DIS, 2))
#     print("RSQ: ", round(RSQ, 2))
    
#     print(df_stat.head())
    
    return df_stat, SS, AIC, DIS, RSQ, Clusters_count
    
    