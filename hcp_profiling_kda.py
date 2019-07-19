# -*- coding: utf-8 -*-
"""
Created on Wed Feb 20 10:30:50 2019
@author: 713538

"""

# =============================================================================
# Solution approach and rationale:-
# Cross Industry Standard Process for Data Mining (CRISP–DM) framework.It involves a series of steps:
# 	1.Business understanding
# 	2.Data understanding
# 	3.Data Preparation & EDA
# 	4.Model Building
# 	5.Model Evaluation
#  6.Model Validation
# 	7.Model Deployment
# 
# =============================================================================

import numpy as np #linear algebra
import pandas as pd #import pandas
import matplotlib.pyplot as plt #eda
#from scipy import stats #imputing missing values
import seaborn as sns # the commonly used alias for seaborn is sns
from sklearn.model_selection import train_test_split
import calendar


######### Business Understanding #############
#Profiling of HCP professionls i.e. understanding the key factors 
#which are highly drive,  or reseaons behind their prescribe drugs or in other words to know the impact of the
#key driving factors for actual sales of drugs.

############# Data Understanding ###################


## imporitng the dataset ########
xls = pd.ExcelFile('mendel_core_data.xlsx')
mendel_data_log = pd.read_excel(xls,"data_log")
#checking the types of data
mendel_data_log.info()
mendel_data_log.head()


##### Deleting columns which are not required ########
#Droping Prob_Score through index bcz it is not accessiblr throgh labels
mendel_data_log.drop(mendel_data_log.columns[18], axis=1,inplace=True)
mendel_data_log.drop(['Rep_name','Rep_Email','Targeting Criteria_1','Targeting Criteria_8','Targeting Criteria_9',
                      'Targeting Criteria_13','Segment','Target_Score_1','Target_Score_2','Target_Score_3',
                      'Target_Score_4','Target_Score_5','Target_Score_6','Target_Score_7',
                      'Target_Score_8','Target_Score_9','Target_Score_10','Target_Score_11','Target_Score_12','DataInput_DataSerial'], axis=1,inplace=True )
#Renaming The columns 
#Rep_ID -> rep_id
# Health Group  -> health_grp
# Account Name -> account_name
# Targeting Criteria_2 -> account_relation
# Targeting Criteria_3 -> injection_potential
# Targeting Criteria_4 -> pal (Product_Adoption_Ladder)
# Targeting Criteria_5 -> competitive_situation
# Targeting Criteria_6 -> clinical_mindset
# Targeting Criteria_7 -> value_perception
# Targeting Criteria_10 -> patients_treated_with_competitive_drug  #VS
# Targeting Criteria_11 -> patients_treated_with_selling_drug      #syngersire
# Targeting Criteria_12 -> competitive_drug_market_penitration (patients_treated_with_competitive_drug/total_potential_value)
# Total Potential Value -> total_potential_value 
# Actual Sales -> actual_sales(This is the Dependent Variable(y= mx+c))
#Taget Sales -> target_sales
#Market Share in Account ->  market_share_in_account ((patients_treated_with_selling_drug/patients_treated_with_competitive_drug))
# % of Territory Potential Sales -> percentage_territory_potential_sales
# % of Territory Actual Sales  ->  percentage_territory_actual_sales
# Territory Quota ->  territory_quota
# Target Sales.1 ->  target_sales_quota
# Territory Quota Attainment ->  territory_quota_attainment (target_sales_quota/territory_quota)
# DataInput_TimeStamp -> input_timestamp
mendel_data_log.info()
mendel_data_log.columns = ['rep_id','health_grp','account_name',
                           'account_relation','injection_potential','pal','competitive_situation',
                           'clinical_mindset','value_perception','patients_treated_with_competitive_drug',
                           'patients_treated_with_selling_drug','competitive_drug_market_penitration',
                           'total_potential_value','actual_sales','target_sales',
                           'market_share_in_account','percentage_territory_potential_sales',
                           'percentage_territory_actual_sales','territory_quota','target_sales_quota',
                           'territory_quota_attainment','input_timestamp']
#writting this as a final data set in local for reference
#mendel_data_log.to_csv("mendel_final_dataset.csv", index = False)
mendel = pd.DataFrame(mendel_data_log)
mendel.info()
mendel.head(5)
mendel.tail()
mendel.describe()


############ Data Preparation #####################

#droping the row which has unneccesary value in all the columns
mendel = mendel[ (mendel['account_relation'] != 'Targeting Criteria_2') & (mendel['injection_potential'] != 'Targeting Criteria_3') 
                & ( mendel['pal'] != 'Targeting Criteria_4') & ( mendel['competitive_situation'] != 'Targeting Criteria_5' )
                & (mendel['clinical_mindset'] != 'Targeting Criteria_6') & (mendel['value_perception'] != 'Targeting Criteria_7') ]
#converting object/ character/ string  data type to numeric wherever required.
cols_num = ['patients_treated_with_competitive_drug','patients_treated_with_selling_drug',
                        'competitive_drug_market_penitration',
                           'total_potential_value','actual_sales','target_sales',
                           'market_share_in_account','percentage_territory_potential_sales',
                           'percentage_territory_actual_sales','territory_quota','target_sales_quota',
                           'territory_quota_attainment']
mendel[cols_num] = mendel[cols_num].apply(pd.to_numeric, errors='coerce', axis=1)
#cols_cat = ['rep_id','health_grp','account_name','account_relation','injection_potential','pal','competitive_situation','clinical_mindset','value_perception']
#converting to date time object and extracting month and weekdays
mendel.input_timestamp =  pd.to_datetime(mendel.input_timestamp)
mendel['month'] = mendel['input_timestamp'].dt.month
mendel['month']= mendel['month'].apply(lambda x: calendar.month_abbr[x])
mendel['day'] = mendel['input_timestamp'].dt.weekday_name

mendel.info()
#since strings data types have variable length, it is by default stored as object dtype. If you want to store them as string type, you can do something like this.
#cols_cat = ['account_relation','injection_potential','pal','competitive_situation','clinical_mindset','value_perception']

#converting all the categorical values in data frame to lower case
mendel = mendel.apply(lambda x: x.str.lower() if(x.dtype == 'object') else x)

#checking for duplicate observations and droping duplicates
dup_med = mendel.duplicated() #will return a sereis of boolean
mendel = mendel.drop_duplicates()

#removing a particular pattern of string in Product_Adoption_Ladder
mendel['pal'] = mendel['pal'].str.split('(').str[0]
mendel['pal'] = mendel['pal'].str.replace(' ', '')
mendel['injection_potential'] = mendel['injection_potential'].str.replace(' ', '')
mendel['competitive_situation'] = mendel['competitive_situation'].str.replace(' ', '')

# Total Number of missing values or NAN in each column & #percentage of missing values each column
mendel.isnull().sum()
round((mendel.isnull().sum()/len(mendel))*100,2)


#----------------------------------------------------------------------------------------------------------------------------------------------------------------
#outlier treatment for numbneric values with a UDF centile
def centile(mendel,num_var):
    var_quantile =  mendel[num_var].quantile(np.arange(0,1.01,0.01))
    print(var_quantile)
    

def outlier_treatment(num_outlier_treat_var, value):
    mendel[num_outlier_treat_var] = np.where(mendel[num_outlier_treat_var] > value ,value ,mendel[num_outlier_treat_var])
    

#-----------------------------------------------------------------------------------------------------------------------------------------------------------------
#total_potential_value





#outlier treatment & handeling missing values
#set a seaborn style of your taste
sns.set_style("whitegrid")
sns.boxplot(mendel['total_potential_value'])
centile(mendel,'total_potential_value')
outlier_treatment('total_potential_value',600)

#handeling missing values of total_potential_value
mendel["total_potential_value"] = mendel["total_potential_value"].ffill().bfill()
mendel["total_potential_value"].describe()

#target_sales

#outlier treatment & handeling missing values
sns.boxplot(mendel['target_sales'])
centile(mendel,'target_sales')
outlier_treatment('target_sales',165)
mendel["target_sales"].describe()
mendel.isnull().sum()
mendel["target_sales"] = mendel["target_sales"].ffill().bfill()


#actual_sales
#outlier treatment & handeling missing values

sns.boxplot(mendel['actual_sales'])
centile(mendel,'actual_sales')
outlier_treatment('actual_sales',134.00)
mendel["actual_sales"].describe()
mendel["actual_sales"] = mendel["actual_sales"].ffill().bfill()
mendel.isnull().sum()


#competitive_drug_market_penitration
#outlier treatment & handeling missing values

sns.boxplot(mendel['competitive_drug_market_penitration']) #no outlier
centile(mendel,'competitive_drug_market_penitration') #no outlier
mendel["competitive_drug_market_penitration"].describe()
mendel["competitive_drug_market_penitration"] = mendel["competitive_drug_market_penitration"].ffill().bfill()
mendel.isnull().sum()

#patients_treated_with_competitive_drug
#outlier treatment & handeling missing values

sns.boxplot(mendel['patients_treated_with_competitive_drug']) 
centile(mendel,'patients_treated_with_competitive_drug') 
outlier_treatment('patients_treated_with_competitive_drug',225.20)
mendel["patients_treated_with_competitive_drug"].describe()
mendel["patients_treated_with_competitive_drug"] = mendel["patients_treated_with_competitive_drug"].ffill().bfill()
mendel.isnull().sum()

#patients_treated_with_selling_drug
#outlier treatment & handeling missing values

sns.boxplot(mendel['patients_treated_with_selling_drug']) 
centile(mendel,'patients_treated_with_selling_drug') 
outlier_treatment('patients_treated_with_selling_drug',85.60)
mendel["patients_treated_with_selling_drug"].describe()
mendel["patients_treated_with_selling_drug"] = mendel["patients_treated_with_selling_drug"].ffill().bfill()
mendel.isnull().sum()

#accout_relation
##handeling missing values  with mode.

sns.countplot(x= mendel['account_relation'], data = mendel)
((mendel['account_relation'].value_counts())/len(mendel))*100
mendel = mendel.fillna(mendel['account_relation'].value_counts().index[0])
mendel["account_relation"].describe()
#mendel['account_relation'] = mendel.fillna(mendel['account_relation'].ffill().bfill())
mendel.info()
mendel.isnull().sum()


#only outlier check
#market share in account
sns.boxplot(mendel['market_share_in_account']) 
outlier_treatment('market_share_in_account',0.88)
centile(mendel,'market_share_in_account') 
mendel["market_share_in_account"].describe()

#percentage_territory_potential_sales
sns.boxplot(mendel['percentage_territory_potential_sales']) 
centile(mendel,'percentage_territory_potential_sales') 
outlier_treatment('percentage_territory_potential_sales',0.066)
mendel.isnull().sum()

#percentage_territory_actual_sales
sns.boxplot(mendel['percentage_territory_actual_sales']) 
centile(mendel,'percentage_territory_actual_sales') 
outlier_treatment('percentage_territory_actual_sales',0.068966)
mendel.isnull().sum()

#territory_quota

sns.boxplot(mendel['territory_quota']) 
centile(mendel,'territory_quota') 
outlier_treatment('territory_quota',3200)
mendel.isnull().sum()

#target_sales_quota
sns.boxplot(mendel['target_sales_quota'])
centile(mendel,'target_sales_quota') 
outlier_treatment('target_sales_quota',1047)
mendel.isnull().sum()

#territory_quota_attainment
sns.boxplot(mendel['territory_quota_attainment'])
centile(mendel,'territory_quota_attainment') 
outlier_treatment('territory_quota_attainment',1)
mendel.isnull().sum()

#derived metrics

###selling_drug_market_penitration_in_account <- patients_treated_with_selling_drug/total_potential_value for each account
mendel['selling_drug_market_penitration'] = mendel['patients_treated_with_selling_drug']/mendel['total_potential_value']
mendel['selling_drug_market_penitration'].isnull().sum() #185 Missing values

#outlier tratment and handeling missing values for selling_drug_market_penitration
sns.boxplot(mendel['selling_drug_market_penitration']) 
centile(mendel,'selling_drug_market_penitration')
outlier_treatment('selling_drug_market_penitration',0.397500)
mendel["selling_drug_market_penitration"].describe()
mendel["selling_drug_market_penitration"] = mendel["selling_drug_market_penitration"].ffill().bfill()
mendel.isnull().sum()

#var_quantile =  mendel['territory_quota_attainment'].quantile(np.arange(0,1.01,0.01))

########### EDA and Derived Metrics ###########
# Univariate analysis:-  frequency and disctribution plot of each categorical and numeric variables respectively

# subplots
#freq. distribution of categorical variables
# subplot 1
plt.subplot(2, 3, 1)
plt.title('account_relation')
sns.countplot(mendel['account_relation'])
# subplot 2
plt.subplot(2,3, 2)
plt.title('injection_potential')
sns.countplot(mendel['injection_potential'])

# subplot 3
plt.subplot(2, 3, 3)
plt.title('Product_Adoption_Ladder')
sns.countplot(mendel['pal'])

# subplot 4
plt.subplot(2, 3, 4)
plt.title('competitive_situation')
sns.countplot(mendel['competitive_situation'])

# subplot 5
plt.subplot(2, 3, 5)
plt.title('clinical_mindset')
sns.countplot(mendel['clinical_mindset'])

# subplot 6
plt.subplot(2, 3, 6)
plt.title('value_perception')
sns.countplot(mendel['value_perception'])
plt.show()



#The columns are positioned over a label that represents a categorical variable.The height of the column indicates the size of the group defined by the column label.

#-------------------------------------------------------------------------------------------------------------------------------------------------------------
#lets define a User defined Fnction to plot actual_sales_percentage across categorical variables
def plot_cat(cat_var):  
    sns.set_style("whitegrid")
    actual_sales_perc_cat =  pd.DataFrame(((mendel.groupby(cat_var).actual_sales.sum())/ np.sum(mendel['actual_sales']) )*100)
    actual_sales_perc_cat.columns = ['actual_sales_percentage']
    plt.title('Actual Sales Percentage Across Category')
    sns.barplot(actual_sales_perc_cat.index , y='actual_sales_percentage', data= actual_sales_perc_cat)
    plt.show()

#-------------------------------------------------------------------------------------------------------------------------------------------------------------

plot_cat('account_relation')
plot_cat('injection_potential')
plot_cat('pal')
plot_cat('competitive_situation')
plot_cat('clinical_mindset')
plot_cat('value_perception')
plot_cat('month')
plot_cat('day')

#-------------------------------------------------------------------------------------------------------------------------------------------------------------
#lets define a User defined Fnction to plot actual_sales across numeric variables
def plot_num(num_var): 
    
    #scatter plor to see how to numeric varibales are related to actual_sales
    sns.set(style="darkgrid", color_codes=True)
    sns.jointplot(num_var, y='actual_sales', data= mendel)
    plt.show()

#--------------------------------------------------------------------------------------------------------------------------------------------------------------

mendel.info()
plot_num('patients_treated_with_competitive_drug')
plot_num('patients_treated_with_selling_drug')
plot_num('competitive_drug_market_penitration')
plot_num('total_potential_value')
plot_num('target_sales')
plot_num('market_share_in_account')
plot_num('percentage_territory_potential_sales')
plot_num('percentage_territory_actual_sales')
plot_num('territory_quota')
plot_num('target_sales_quota')
plot_num('territory_quota_attainment')
plot_num('selling_drug_market_penitration')

#corelation matrix between numeric variables & corelation between numeric variables

# using mendel.corr()
cor = pd.DataFrame(mendel[cols_num].corr())
round(cor, 2)
# figure size # heatmap
plt.figure(figsize=(10,8))
sns.heatmap(cor, cmap="YlGnBu", annot=True)
# pairplot
sns.pairplot(cor)
plt.show()




############# Scaling , Sampling and Dummy variables creation and feature selection for model#####################################

#--------------------------------------------------------------------------------------------------------------------------------
#defining a normalisation function : ReScaling
def normalize (x): 
    return ( (x-np.mean(x))/ (np.max(x) - np.min(x)))

#--------------------------------------------------------------------------------------------------------------------------------                                                                                   
#selecting numerics columns only after judging from EDA which are looks significant
cols_num_scaled = ['patients_treated_with_competitive_drug','patients_treated_with_selling_drug',
                        'competitive_drug_market_penitration','total_potential_value',
                    'target_sales','market_share_in_account','selling_drug_market_penitration' ]

num_mendel = mendel.loc[:,cols_num_scaled]
# Normalizing and standadization of numeric variables
num_mendel = num_mendel.apply(normalize)

#selecting catgorical  variables
cols_cat = ['account_relation','injection_potential','pal','competitive_situation','clinical_mindset','value_perception','month','day']
cat_mendel = mendel.loc[:,cols_cat]
#Creating dummy varibles for categorical variables

# we can use drop_first = True to drop the first column from dummy dataframe.
cat_mendel_dummy = pd.get_dummies(cat_mendel,drop_first=True)

#creating the final data set which wre going to throw to the model
#adding dummy data frames of categorical varibles and normilized data frames of numeric varibles
mendel_final_df = pd.concat([cat_mendel_dummy,num_mendel,mendel['actual_sales']],axis=1)
mendel_final_df.info()

#mendel_final_df.to_csv("mendel_final_df.csv", index = False)

#spliting the data set into test and train
#random_state is the seed used by the random number generator, it can be any integer.
# Putting feature variables to X
X = mendel_final_df.loc[:, mendel_final_df.columns !='actual_sales' ]
# Putting dependent or response  variable to y
y = mendel_final_df['actual_sales']
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.7 ,test_size = 0.3, random_state= 81)




###################### Model Building ############################


# Importing RFECV and LinearRegression
from sklearn.feature_selection import RFECV
from sklearn.linear_model import LinearRegression
import statsmodels.api as sm          # Importing statsmodels
#from sklearn.model_selection import StratifiedKFold

lm = LinearRegression()
rfe_cv = RFECV(lm, step= 1 , cv = 5)
rfe_cv = rfe_cv.fit(X_train, y_train)
print(rfe_cv.support_)                      # Printing the boolean results
print(rfe_cv.ranking_)  
print(rfe_cv.n_features_)
rfe_cv.grid_scores_

# Creating X_test dataframe with RFECV selected variables
col_rfe_cv = X_train.columns[rfe_cv.support_]
X_train_rfe_cv = X_train[col_rfe_cv]
print(col_rfe_cv)

##model_1

#Adding a constant column to our dataframe
X_train_rfe_cv = sm.add_constant(X_train_rfe_cv)   
# create a first fitted model
lm_1 = sm.OLS(y_train,X_train_rfe_cv).fit()
#Let's see the summary of our first linear model
print(lm_1.summary())

#--------------------------------------------------------------------------------------------------
# UDF for calculating vif value
def vif_cal(input_data, dependent_col):
    vif_df = pd.DataFrame( columns = ['Var', 'Vif'])
    x_vars=input_data.drop([dependent_col], axis=1)
    xvar_names=x_vars.columns
    for i in range(0,xvar_names.shape[0]):
        y=x_vars[xvar_names[i]] 
        x=x_vars[xvar_names.drop(xvar_names[i])]
        rsq=sm.OLS(y,x).fit().rsquared  
        vif=round(1/(1-rsq),2)
        vif_df.loc[i] = [xvar_names[i], vif]
    return vif_df.sort_values(by = 'Vif', axis=0, ascending=False, inplace=False)


df_rfe_cv =  pd.concat([X_train_rfe_cv, y_train], axis = 1)
df_rfe_cv.info()

#we are droping the constant bcz we have added constant for stat models.bcz stat models doesnot automatically add a consatnt.
vif_cal(input_data= df_rfe_cv.drop(["const"], axis=1) ,  dependent_col = "actual_sales")
# Dropping highly correlated variables and insignificant variables
#droping month_oct bcz VIF> 3.00
X_train_rfe_cv = X_train_rfe_cv.drop('month_oct', 1)
#----------------------------------------------------------------------------------------------------
X_train_rfe_cv.info()
##model_2
# Create a second fitted model
lm_2 = sm.OLS(y_train,X_train_rfe_cv).fit()
print(lm_2.summary())
vif_cal(input_data= df_rfe_cv.drop(["const","month_oct"], axis=1) ,  dependent_col = "actual_sales")
#all VIF under 3 so from now on we will check only P values.
print(lm_2.summary())
#droping patients_treated_with_competitive_drug
X_train_rfe_cv = X_train_rfe_cv.drop('patients_treated_with_competitive_drug', 1)

#model_3 

lm_3 = sm.OLS(y_train,X_train_rfe_cv).fit()
print(lm_3.summary())
#vif_cal(input_data= df_rfe_cv.drop(["const","patients_treated_with_competitive_drug","percentage_territory_potential_sales"], axis=1) ,  dependent_col = "actual_sales")

#droping pal_mixeduser
X_train_rfe_cv = X_train_rfe_cv.drop('pal_mixeduser', 1)

#model_4

lm_4 = sm.OLS(y_train,X_train_rfe_cv).fit()
print(lm_4.summary())

X_train_rfe_cv.info()
#droping pal_pro-competitor
X_train_rfe_cv = X_train_rfe_cv.drop('pal_pro-competitor', 1)

## model_5
lm_5 = sm.OLS(y_train,X_train_rfe_cv).fit()
print(lm_5.summary())
#vif_cal(input_data= df_rfe_cv.drop(["const","territory_quota_attainment","territory_quota","target_sales_quota","patients_treated_with_competitive_drug"], axis=1) ,  dependent_col = "actual_sales")

# droping pal_trialist
X_train_rfe_cv = X_train_rfe_cv.drop('pal_trialist', 1)

##model_6
lm_6 = sm.OLS(y_train,X_train_rfe_cv).fit()
print(lm_6.summary())

# droping pal_trialist
X_train_rfe_cv = X_train_rfe_cv.drop('competitive_drug_market_penitration', 1)

##model_7
lm_7 = sm.OLS(y_train,X_train_rfe_cv).fit()
print(lm_7.summary())


final_model = sm.OLS(y_train,X_train_rfe_cv).fit()
print(final_model.summary()) #R-squared:0.665


############ Making Prediction ##################

# Now let's use our model to make predictions.
# Creating X_test_6 dataframe by dropping variables from X_test
X_test_rfe_cv = X_test[col_rfe_cv]

X_test_rfe_cv = X_test_rfe_cv.drop(["month_oct","patients_treated_with_competitive_drug","pal_mixeduser","pal_pro-competitor","competitive_drug_market_penitration","pal_trialist"], axis=1)

X_test_rfe_cv.info()
# Adding a constant variable 
X_test_rfe_cv = sm.add_constant(X_test_rfe_cv)

# Making predictions
y_pred = final_model.predict(X_test_rfe_cv)




######################### Model Evaluation #########################



# Actual vs Predicted
c = [i for i in range(0,358,1)]
fig = plt.figure()
plt.plot(c,y_test, color="blue", linewidth=2.5, linestyle="-")  #Plotting Actual in blue
plt.plot(c,y_pred, color="red",  linewidth=2.5, linestyle="-")  #Plotting predicted in red
fig.suptitle('Actual and Predicted', fontsize=20)              # Plot heading 
plt.xlabel('Index', fontsize=18)                               # X-label
plt.ylabel('actual_sales', fontsize=16) 


# Error terms
c = [i for i in range(0,358,1)]
fig = plt.figure()
plt.plot(c,y_test - y_pred, color="blue", linewidth=2.5, linestyle="-")
fig.suptitle('Error Terms', fontsize=20)              # Plot heading 
plt.xlabel('Index', fontsize=18)                      # X-label
plt.ylabel('residuals', fontsize=16)   

# Plotting the error terms to understand the distribution.
fig = plt.figure()
sns.distplot((y_test - y_pred),bins=50)
fig.suptitle('Residuals Distribution', fontsize=20)       # Plot heading 
plt.xlabel('y_test-y_pred', fontsize=18)                  # X-label
plt.ylabel('Index', fontsize=16) 

# Plotting y_test and y_pred to understand the spread.
fig = plt.figure()
plt.scatter(y_test,y_pred)
fig.suptitle('y_test vs y_pred', fontsize=20)              # Plot heading 
plt.xlabel('y_test', fontsize=18)                          # X-label
plt.ylabel('y_pred', fontsize=16)                          # Y-label


# Now let's check the Root Mean Square Error of our model.
from sklearn import metrics
print('RMSE :', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))



#check for linear regression assumtions
#1. Linearity and Additive
#resuduauls vs fitted values
fig = plt.figure()
plt.scatter(y_test - y_pred,y_pred)
fig.suptitle('residuals vs fitted values', fontsize=20) 
plt.xlabel('y_pred', fontsize=18)                          # X-label
plt.ylabel('residuals', fontsize=16)  

########## Residual Analysis ####################################################

#Breusch-Pagan Lagrange Multiplier test for heteroscedasticity : sigma_i = sigma * f(alpha_0 + alpha z_i)
#H0 : the null hypothesis that heteroskedasticity is not present (i.e. homoskedastic)
#this is the most important test 
from statsmodels.compat import lzip
from statsmodels.stats import diagnostic as diagnstc
name = ['Lagrange multiplier statistic', 'p-value','f-value', 'f p-value']
test = diagnstc.het_breuschpagan(final_model.resid, final_model.model.exog)
lzip(name, test)


### independence test --- >  lag more than 1-- >  H0: no auto corelation
name = ['F statistic', 'p-value']
test = diagnstc.het_goldfeldquandt(final_model.resid, final_model.model.exog)
lzip(name, test)

###linearity test : Harvey colier test  
#name = ['t value', 'p value']
#test = diagnstc.linear_harvey_collier(final_model)
#lzip(name, test)


############## Test OF Nomality ##################

#normality test  : jarque bera
import statsmodels.stats.api as sms
name = ['Jarque-Bera', 'Chi^2 two-tail prob.', 'Skew', 'Kurtosis']
test = sms.jarque_bera(final_model.resid)
lzip(name, test)


#normality Test : Q-Q plot
# QQ Plot
residuals =  y_test - y_pred
from numpy.random import seed
from statsmodels.graphics.gofplots import qqplot
#from matplotlib import pyplot
seed(81)
qqplot(residuals, line='s')
plt.show()

# Normality test : Shapiro-Wilk Test
from scipy.stats import shapiro
seed(36)
stat, p = shapiro(residuals)
print('Statistics=%.3f, p=%.3f' % (stat, p))
# interpret
alpha = 0.05
if p > alpha:
	print('Sample looks Gaussian (fail to reject H0)')
else:
	print('Sample does not look Gaussian (reject H0)')
    

# Anderson-Darling Test : test of normality
from scipy.stats import anderson
# seed the random number generator
seed(36)
result = anderson(residuals)
print('Statistic: %.3f' % result.statistic)
p = 0
for i in range(len(result.critical_values)):
	sl, cv = result.significance_level[i], result.critical_values[i]
	if result.statistic < result.critical_values[i]:
		print('%.3f: %.3f, data looks normal (fail to reject H0)' % (sl, cv))
	else:
		print('%.3f: %.3f, data does not look normal (reject H0)' % (sl, cv))


#-------------------------------------------------------------------------------------------------------------
########### Random Forest Regressor with Hyper parameters tuned by GridSearchCV through RandomedSearchCV#########
#--------------------------------------------------------------------------------------------------------------
from pprint import pprint
from sklearn.model_selection import RandomizedSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
rf = RandomForestRegressor(random_state = 81)
# Look at parameters used by our current forest
print('Parameters currently in use:\n')
pprint(rf.get_params())
# Hyper parametrs tuning random serach with cross validation 
# Number of trees in random forest
n_estimators = [int(x) for x in np.linspace(start = 200, stop = 2000, num = 10)]
# Number of features to consider at every split
max_features = ['auto', 'sqrt']
# Maximum number of levels in tree
max_depth = [int(x) for x in np.linspace(10, 110, num = 11)]
max_depth.append(None)
# Minimum number of samples required to split a node
min_samples_split = [36]
# Minimum number of samples required at each leaf node
min_samples_leaf = [16]
# Method of selecting samples for training each tree
bootstrap = [True]

# Create the random grid
random_grid = {'n_estimators': n_estimators,
               'max_features': max_features,
               'max_depth': max_depth,
               'min_samples_split': min_samples_split,
               'min_samples_leaf': min_samples_leaf,
               'bootstrap': bootstrap}

pprint(random_grid)

# Use the random grid to search for best hyperparameters
# First create the base model to tune
rf = RandomForestRegressor(random_state = 81)
# Random search of parameters, using 4 fold cross validation, 
# search across 100 different combinations, and use all available cores
rf_random = RandomizedSearchCV(estimator=rf, param_distributions=random_grid,
                              n_iter = 100, scoring='neg_mean_absolute_error', 
                              cv = 5, verbose=2, random_state=81, n_jobs=-1,
                              return_train_score=True)

# Fit the random search model
rf_random.fit(X_train, y_train)
rf_random.best_params_


#----------------------------------------------------------------------------------------------------------------
# RF model evaluation function : accuracy and rsme
def evaluate(model, X_test, y_test):
    predictions = model.predict(X_test)
    rsme = np.sqrt(metrics.mean_squared_error(y_test, predictions))    
    accuracy = 100 - rsme
    pprint('Model Performance')
    pprint('root mean sq error: {:0.4f} degrees.'.format(rsme))
    pprint('Accuracy = {:0.2f}%.'.format(accuracy)) 
    return accuracy

#Variable Importances & visualizastion
def important_features(model,features):
    feature_list = list(features.columns) #Saving feature names for later use
    importances = list(model.feature_importances_) # Get numerical feature importances
    feature_importances = [(features, round(importance, 2)) for features, importance in zip(feature_list, importances)] # List of tuples with variable and importance
    feature_importances = sorted(feature_importances, key = lambda x: x[1], reverse = True) # Sort the feature importances by most important first
    [pprint('Variable: {:20} Importance: {}'.format(*pair)) for pair in feature_importances]# Print out the feature and importances 
    x_values = list(range(len(importances))) # list of x locations for plotting
    plt.bar(x_values, importances, orientation = 'vertical') # Make a bar chart
    plt.xticks(x_values, feature_list, rotation='vertical') # Tick labels for x axis
    plt.ylabel('Importance'); plt.xlabel('Variable'); plt.title('Variable Importances') # Axis labels and title
   
#---------------------------------------------------------------------------------------------------------------
#evaluate the best random search model    
best_random = rf_random.best_estimator_
random_accuracy = evaluate(best_random, X_test, y_test)

#Variable Importances & visualizastion
important_features(best_random,X_test)

#We can now perform grid search building on the result from the random search. 
#We will test a range of hyperparameters around the best values returned by random search.
# Create the parameter grid based on the results of random search 
param_grid = {
    'bootstrap': [True],
    'max_depth': [72,81,90],
    'max_features': [5],
    'min_samples_leaf': [9],
    'min_samples_split': [36],
    'n_estimators': [800]
}



# Instantiate the grid search model
grid_search_final = GridSearchCV(estimator = rf, param_grid = param_grid, 
                                 cv = 4, n_jobs = -1, verbose = 2, return_train_score=True)
grid_search_final.fit(X_train, y_train)



grid_search_final.best_params_
best_grid = grid_search_final.best_estimator_
grid_accuracy = evaluate(best_grid, X_test, y_test)
important_features(best_grid,X_test)
y_pred_rf = best_grid.predict(X_test)
#final_model = best_grid
#
## Use sklearn to export the tree 
#from sklearn.tree import export_graphviz
#
## Write the decision tree as a dot file
#visual_tree = final_model.estimators_[12]
#export_graphviz(visual_tree, out_file = 'images/best_tree.dot', feature_names = important_feature_names, 
#                precision = 2, filled = True, rounded = True, max_depth = None)
#
#

#predictions = best_random.predict(X_test)
#errors = abs(predictions - y_test)
#  mape = 100 * np.mean(errors / y_test)
#  accuracy = 100 - mape#  
#  print(mape)
#pprint(errors)# best_random.score
 
# # list of x locations for plotting
#x_values = list(range(len(importances)))
#
## Make a bar chart
#plt.bar(x_values, importances, orientation = 'vertical')
#
## Tick labels for x axis
#plt.xticks(x_values, feature_list, rotation='vertical')
#
## Axis labels and title
#plt.ylabel('Importance'); plt.xlabel('Variable'); plt.title('Variable Importances');
#--------------------------------------------------------------------------------------------------------------------------------
##percentage of sales accros categor
#    
#sns.lmplot(y='actual_sales', x='percentage_territory_actual_sales', data=mendel)

#xx = ((mendel.groupby('injection_potential').actual_sales.sum())/ sum(mendel['actual_sales']) )*100
#
#y=  sum(mendel['actual_sales'])
#
#
#
#xxy = 
#sns.distplot(xx['actual_sales'])
#
#var_quantile =  mendel['patients_treated_with_selling_drug'].quantile(np.arange(0,1.01,0.01))
##total_potential_value
#
#mendel.total_potential_value.describe()
#sns.distplot(mendel['total_potential_value'].notnull(),hist=True,bins=100)
#
#sns.distplot(xx['actual_sales'],rug = True)
#
##    errors = abs(predictions - y_test)
#    mape = 100 * (np.mean(errors / y_test))
#
#help(pd.qcut)
#
#pd.qcut(mendel['total_potential_value'], .05)
#
#
#
#np.percentile(mendel['total_potential_value'], 100 * 0.95)
#
#z = np.abs(stats.zscore((mendel['total_potential_value'].notnull())))
#print(z)
#
#mendel.total_potential_value.quantile([.05,.95])
##market_share_in_account
#mean_value = np.mean(mendel['market_share_in_account'])
#mendel['market_share_in_account'].describe()
#print(mean_value)
#
#
#
##market_share_in_account
#mean_value = np.mean(mendel['actual_sales'])
#mendel['actual_sales'].describe()
#print(mean_value)
##distriution of market_share_in_account
## rug = True
## plotting only a few points since rug takes a long while
#sns.distplot(mendel['actual_sales'].notnull(), rug=True)
##seaborn.distplot(data['alcconsumption'].notnull(),hist=True,bins=100)
#
##mendel['percentage_territory_potential_sales'] = mendel.fillna(mendel['percentage_territory_potential_sales'].)
##
##mendel.health_grp.describe()
#
#
#
#from sklearn.preprocessing import Imputer
#imp = Imputer(missing_values='NaN', strategy='most_frequent', axis=0)
##imp.fit(mendel['account_relation'])
##mendel['account_relation'] = imp.transform(mendel['account_relation'])
###mendel.to_csv("mendel_final_dataset.csv", index = False)
#mendel['pal'] = mendel['pal'].map(lambda x: x.lstrip('(-').rstrip('aAbBcC'))












