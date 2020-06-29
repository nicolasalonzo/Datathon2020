import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
from graphviz import Source
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, cross_val_score
from sklearn.model_selection import KFold
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet,  MultiTaskLasso , LassoLars, BayesianRidge, ARDRegression, LogisticRegression,SGDRegressor
from sklearn.metrics import mean_absolute_error,mean_squared_error
from sklearn.metrics import make_scorer , r2_score
from sklearn.utils import shuffle
from sklearn.svm import SVR, LinearSVR
from datetime import timedelta
from sklearn.neural_network import MLPClassifier, MLPRegressor

from sklearn.compose import TransformedTargetRegressor

from sklearn import cluster, datasets, mixture

from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from itertools import cycle, islice


from missingpy import MissForest
#from missingpy import KNNImputer




#https://developers.google.com/public-data/docs/canonical/countries_csv


from matplotlib import pyplot as plt

def evaluate(model, X, Y):
    strat_k_fold = KFold(n_splits=3, shuffle=True, random_state=2020)
    mae = make_scorer(mean_absolute_error)
    return np.mean(cross_val_score(model, X, Y, scoring = mae, cv = strat_k_fold))


def log_transform(x):
    return np.log1p(x)
def exponential_transform(x):
    return np.expm1(x)

def make_model_transformation(model):
    modelT = TransformedTargetRegressor(regressor=model,
                                   func=log_transform,
                                   inverse_func=exponential_transform)
    return modelT


def scalar_transform(x):
    #print(x)
    scaler = StandardScaler()
    #scaler.fit(x)
    return scaler.fit_transform([x])[0]


def min_transform(x):
    #print(x)
    scaler = MinMaxScaler()
    #scaler.fit(x)
    return scaler.fit_transform([x])[0]



#--------------------lectura del dataset-----------------------------------
#

df_mobility = pd.read_csv('Global_Mobility_Report.csv', encoding = 'iso-8859-1', low_memory=False)

df_mobility['date'] = pd.to_datetime(df_mobility['date'])

df_mobility = df_mobility[df_mobility['date'] == pd.to_datetime('2020-02-29')]

df_mobility_temp = pd.DataFrame()

df_mobility_temp['country_region_code'] = df_mobility['country_region_code']
df_mobility_temp['retail_and_recreation'] = df_mobility['retail_and_recreation_percent_change_from_baseline']
df_mobility_temp['grocery_and_pharmacy'] = df_mobility['grocery_and_pharmacy_percent_change_from_baseline']
df_mobility_temp['parks'] = df_mobility['parks_percent_change_from_baseline']
df_mobility_temp['transit_stations'] = df_mobility['transit_stations_percent_change_from_baseline']
df_mobility_temp['workplaces'] = df_mobility['workplaces_percent_change_from_baseline']
df_mobility_temp['residential'] = df_mobility['residential_percent_change_from_baseline']


df_mobility_temp = df_mobility_temp.dropna()

df_mobility = df_mobility_temp

df_mobility = df_mobility.groupby(['country_region_code']).mean()


df_mobility.to_csv('Global_Mobility_Report_process.csv')







df = pd.read_csv('DATA_RETO_2.csv', encoding = 'iso-8859-1')

columns_mean_continent =[
    'population_density',
    'median_age',
    'aged_65_older',
    'gdp_per_capita',
    'extreme_poverty',
    'cvd_death_rate',
    'diabetes_prevalence',
    'female_smokers',
    'male_smokers',
    'hospital_beds_per_thousand',
    'continent'
]

df2 = df[columns_mean_continent].dropna()
df2 = df2.groupby(['continent']).mean()
df2 = df2.reset_index()



for c in columns_mean_continent[:-1]:
    for i, row in df.iterrows():
        #print (i)

        if df.loc[i, c] == 0.0 or  np.isnan(df.loc[i, c]):
            for k in df2.filter(items=[c, 'continent']).values:
                if k[1] == df.loc[i, 'continent']:
                    df.loc[i, c] = k[0]

df = pd.merge(left=df, right=df_mobility, left_on='country_code', right_on='country_region_code',how='left')
#print (df)

#df.head()
#-----------------------------------------------------------------------


#df = pd.read_csv('df_2.csv', encoding = 'iso-8859-1')
df.head()

#exit()

columns_remove = [
'date',
'country',
'country_code',
'iso',

'handwashing_facilities',

'cumulative_cases',

#'latitude',
#'longitude',

#"ten_cases",
#"retail_recreation",
#"grocery_pharmacy",
#"parks",
#"transit_stations",
#"workplace",
#"residential",
'aged_70_older',
]



#df = df.drop(['handwashing_facilities'],axis = 1)
X = df.copy().drop(columns_remove, axis=1)
Y = df.copy()['cumulative_cases']

if 'Unnamed: 0' in X.columns:
    X = X.drop(['Unnamed: 0'], axis=1)


if True:

    #----------------------Relleno de datos faltantes-------------------------------
    cols_with_missings = []
    for col in X.columns:
        if ~(df.dtypes[col]==np.object):
            missings = np.sum(pd.isna(X[col]))
            #print(col, ':', missings)
            if missings:
                cols_with_missings += [col]

    #imp = SimpleImputer(missing_values=np.nan, strategy='most_frequent')
    #imp = SimpleImputer(missing_values=np.nan, strategy='mean')
    imp = KNNImputer(n_neighbors=3, weights='uniform')
    #imp = MissForest()
    
    #print(X[cols_with_missings])
    for col in cols_with_missings:
        #print(col)
        X[col] = imp.fit_transform(X[[col]]).ravel()
    #----------------------------------------------------------------------------




#---------------------------aplicar tranformacion logaritmica-----------------------------------------
if True:
    columns_selections = [
    'population',
    'population_density',
    'median_age',
    'aged_65_older',
    'gdp_per_capita',
    'extreme_poverty',
    'cvd_death_rate',
    'diabetes_prevalence',
    'female_smokers',
    'male_smokers',   
    'hospital_beds_per_thousand',
    'life_expectancy',
    ]
    for c in columns_selections:
        X[c] = log_transform(X[c])


#-----------------------------------------------------------------------------------------------------


X = X.drop(['continent'], axis=1)



#-------------------------------------------------------------------------------------------
X = pd.get_dummies(X, prefix_sep='_')


X = X.drop(['who_region_AFRO'], axis=1)



#-----------------------------------------------------------------------------------------





X.to_csv('train.csv')
Y.to_csv('train_Y.csv')


#------------------bateria de modelos a evaluar --------------------------------------------------

list_models = [
    #{'name':'MLPClassifier','model':MLPClassifier(solver='lbfgs', alpha=1e-5,hidden_layer_sizes=(5, 2), random_state=1)},
    #{'name':'MLPRegressor','model':MLPRegressor(max_iter=2000)},

    {'name':'MLPRegressorOptime','model':MLPRegressor(hidden_layer_sizes=(10,),  activation='relu', solver='adam', alpha=0.001, batch_size='auto',learning_rate='constant', learning_rate_init=0.01, power_t=0.5, max_iter=2000, shuffle=True,random_state=9, tol=0.0001, verbose=False, warm_start=False, momentum=0.9, nesterovs_momentum=True,early_stopping=False, validation_fraction=0.1, beta_1=0.9, beta_2=0.999, epsilon=1e-08)},
    
    {'name':'SVRDefault','model':SVR()},
    {'name':'SVROptim','model':SVR(degree=1e-120, epsilon=1e-200)},
    #{'name':'SVROptim2','model':SVR(C=4, degree=0.001, epsilon=0.01, gamma='scale', kernel='linear')},
    {'name':'SVROptim3','model':SVR(C=4, degree=0.001, epsilon=1.000000000000001e-145, tol=0.001, coef0=1.0000000000000009e-137, max_iter=10000, kernel='rbf')},
    {'name':'SVROptim4','model':SVR(C=3, degree=0.01, epsilon=0.0001, tol=0.01, coef0=0.0001, max_iter=100000, kernel='rbf')},
    {'name':'SVROptim5','model':SVR(C=1000, cache_size=300,gamma=0.005, degree=0.01, epsilon=500, tol=0.01, coef0=0.0001, max_iter=500000, kernel='rbf')},

    #{'name':'SGDRegressorDefault','model':SGDRegressor(loss="squared_loss", penalty=None)},
    
    
    #{'name':'LinearSVR','model':LinearSVR(max_iter=1000,tol=0.0001 )},    
    {'name':'LinearRegressionDefault','model':LinearRegression()},
    {'name':'RidgeDefault','model':Ridge()},


    {'name':'LinearRegression', 'model':LinearRegression()},
    {'name':'Ridge', 'model':Ridge()},
    {'name':'Lasso', 'model':Lasso()},
    {'name':'ElasticNet', 'model':ElasticNet()},


    {'name':'LassoLarsDefault', 'model':LassoLars()},
    {'name':'BayesianRidgeDefault', 'model':BayesianRidge()},
    {'name':'ARDRegressionDefault', 'model':ARDRegression(fit_intercept=True)},
    {'name':'ARDRegression', 'model':ARDRegression(fit_intercept=True, threshold_lambda=10000)},
    {'name':'ARDRegressionOptim1', 'model':ARDRegression(fit_intercept=True,n_iter=100,tol=0.1, alpha_1=1.0000000000000004e-39, alpha_2=0.1, lambda_1=0.1, lambda_2=1.0000000000000004e-35,threshold_lambda=100)},
    {'name':'ARDRegressionOptim2', 'model':ARDRegression(fit_intercept=True,n_iter=100,tol=0.0001, alpha_1=1.0000000000000004e-39, alpha_2=1e-32, lambda_1=1.0000000000000003e-70, lambda_2=0.0001,threshold_lambda=100)},
    

    

    
    
    #{'name':'LogisticRegression', 'model':LogisticRegression(max_iter=300,solver='liblinear')},
]



#--------------------------------------------------------------------------------------------------------------------





#-------------------------- evaluacion de modelos -------------------------------------------
best_model = None




if True:
    
    for model in list_models:
            regressor = model['model']
            name_model = model['name']
            obj_model = model['model']
            
            #print (name_model)
            score = 99999999999999
            model['score'] = score
            
            try:
                modelT = make_model_transformation(obj_model)
                modelT.fit(X, Y)
                print()
                score = evaluate(modelT, X, Y)
                model['score'] = score
                
            except:
                pass    
            print( name_model+ ': mean_absolute_error ' + str(score))
            
    
    newlist = sorted(list_models, key=lambda k: k['score']) 
    best_model = newlist[0]

    #------------------------- guardar resultados ----------------------------------------
    with open('output_score.txt','w') as file_output:
        for model in newlist:
            name_model = model['name']
            obj_model = model['model']
            score_model = model['score']
            file_output.write(name_model + ' : mean_absolute_error : ' + str(score_model)+ '\n')
    #------------------------------------------------------------------------------------

#------------------------------------------------------------------------------------
    
print ('------------------------------------------------------------------')

print()

print ('best model:', best_model['name'], 'with ', best_model['score'])



if False:

    #SVR
    gripParameter = {
        'kernel':['rbf'],
        'degree':[0.1,0.01,0.001,1e-12],
        'gamma':['scale', 'auto'],
        #'shrinkingbool':[True,False],
        'coef0': [0.01,0.001,1e-10],
        'tol': [0.001, 0.0001, 0.00001],
        'C': [ 3, 4, 5,6],
        'epsilon': [ 0.001,0.00001,1e-12],
        #'cache_size': [ default=200],
        #'max_iterint': [ default=-1],
    }
 
    
    #ARDRegression
    gripParameter = {
        'n_iter': [50,100,300],
        'tol': [1e-3, 0.1, 0.01, 1e-12],
        'alpha_1': [1e-6,1e-12, 0.1, 0.01],
        'alpha_2': [1e-6,1e-12, 0.1, 0.01],
        'lambda_1': [1e-6,1e-12, 0.1, 0.01],
        'lambda_2': [1e-6,1e-12, 0.1, 0.01],
        #'compute_score': [False,True],
        'threshold_lambda': [10,100,1000,3000],
        'fit_intercept': [False,True],
    }

    '''
    #SVR2
    gripParameter={
            'C': [0.01, 0.1, 1, 100, 1000,10000],
            'epsilon': [0.00001,0.00005, 0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1, 5, 10, 50,100,500],
            'gamma': [0.00001, 0.0001, 0.001, 0.005, 0.1, 1, 3, 5, 10,50],
            'cache_size':[100,200,300,400],
            'degree':[0.01,0.001,0.0001],
            
            'tol':[0.01,0.001, 0.0001], 
            'coef0':[0.1, 0.01, 0.001,0.0001],
            'max_iter':[100, 1000, 10000],
        }
    '''
    #MLPRegressor
    gripParameter={
        'activation':['relu'],
        'solver':[ 'adam'],
        'learning_rate':['adaptive'],
        'nesterovs_momentum':[False],
        'early_stopping':[False],
        'max_iter':[200, 500, 1000],

        'beta_1':[0.8,0.9,1.0],
        'beta_2':[0.99,0.999,0.9999],
        'epsilon':[1e-7,1e-8,1e-9],
        'random_state':[2020],
    }

    kf = KFold(n_splits=3, shuffle=True, random_state=2020)
    grid1 = GridSearchCV(estimator=MLPRegressor(), param_grid=gripParameter, verbose=10, cv=kf,scoring= 'neg_mean_squared_error')

    '''
    grid_result = grid1.fit(X, Y)
    best_params = grid_result.best_params_
    print(best_params)
    '''

    best_svr = MLPRegressor(**{'random_state':2020,'activation': 'relu', 'beta_1': 0.9, 'beta_2': 0.9999, 'early_stopping': False, 'epsilon': 1e-08, 'learning_rate': 'adaptive', 'max_iter': 5000, 'nesterovs_momentum': False, 'solver': 'adam'})
    print ('best',evaluate(best_svr, X, Y))
    

    grid = make_model_transformation(grid1).regressor
    grid.fit(X, Y)

    print(grid1.best_score_)
    print(grid1.best_estimator_)


print()
print ('------------ best prediction-----------------------')
#best_model = list_models[0]


model= best_model['model']

model = make_model_transformation(model)
print('begin training')
model.fit(X, Y) 
print('ending training')

print('predicting')
pred_y = model.predict(X)

print('evaluating')
score = evaluate(model, X, Y)
print(best_model['name'] + '_test: mean_absolute_error', score)

#print (model.coef_)
print()


print ('')
regr = model.regressor_




print('Coefficients: \n')
for i in range(len(X.columns)):
    print (X.columns[i],':',regr.coef_[i])
print()
# Este es el valor donde corta el eje Y (en X=0)
print('Independent term: \n', regr.intercept_)


X['pred_cumulative_cases'] = pred_y
X['cumulative_cases'] = Y
X.to_csv('pred.csv')



exit()




#print (pred_y)

plt.scatter(X.index, pred_y, c='red', alpha=0.5)
plt.scatter(X.index, Y, c='blue', alpha=0.5)
plt.title('Scatter plot Datathon 2020 ('+ best_model['name'] +')')
plt.figtext(.15, .8, "Score = "+str(best_model['score']))
plt.xlabel('x')
plt.ylabel('y')
plt.show()
