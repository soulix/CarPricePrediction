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
from scipy import stats #imputing missing values
import seaborn as sns # the commonly used alias for seaborn is sns



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
mendel.input_timestamp =  pd.to_datetime(mendel.input_timestamp)
mendel.info()
#since strings data types have variable length, it is by default stored as object dtype. If you want to store them as string type, you can do something like this.
#cols_cat = ['rep_id','health_grp','account_name','account_relation','injection_potential','pal','competitive_situation','clinical_mindset','value_perception']
#converting all the categorical values in data frame to lower case
mendel = mendel.apply(lambda x: x.str.lower() if(x.dtype == 'object') else x)

#checking for duplicate observations and droping duplicates
dup_med = mendel.duplicated() #will return a sereis of boolean
mendel = mendel.drop_duplicates()


#removing a particular pattern of string in Product_Adoption_Ladder
mendel['pal'] = mendel['pal'].str.split('(').str[0]

# Total Number of missing values or NAN in each column & #percentage of missing values each column
mendel.isnull().sum()
round((mendel.isnull().sum()/len(mendel))*100,2)


#----------------------------------------------------------------------------------------------------------------------------------------------------------------
#outlier treatment for numbneric values with a UDF centile
def centile(mendel,num_var):
    var_quantile =  mendel[num_var].quantile(np.arange(0,1.01,0.01))
    print(var_quantile)
#-----------------------------------------------------------------------------------------------------------------------------------------------------------------
#total_potential_value]

#outlier treatment & handeling missing values
#set a seaborn style of your taste
sns.set_style("whitegrid")
sns.boxplot(mendel['total_potential_value'])
centile(mendel,'total_potential_value')
mendel["total_potential_value"] = np.where(mendel["total_potential_value"] > 600, 600 ,mendel["total_potential_value"])
#ffill() will propagate the closest value forwards through nans and bfill() will propagate the closest value backwards through nans.
#later we will do it with fancyimpute
#handeling missing values of total_potential_value
mendel["total_potential_value"] = mendel["total_potential_value"].ffill().bfill()
mendel["total_potential_value"].describe()

#target_sales

#outlier treatment & handeling missing values
sns.boxplot(mendel['target_sales'])
centile(mendel,'target_sales')
mendel["target_sales"] = np.where(mendel["target_sales"] > 165 , 165  ,mendel["target_sales"])
mendel["target_sales"].describe()
mendel.isnull().sum()
mendel["target_sales"] = mendel["target_sales"].ffill().bfill()


#actual_sales
#outlier treatment & handeling missing values

sns.boxplot(mendel['actual_sales'])
centile(mendel,'actual_sales')
mendel["actual_sales"] = np.where(mendel["actual_sales"] > 133 , 133 ,mendel["actual_sales"])
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
mendel["patients_treated_with_competitive_drug"] = np.where(mendel["patients_treated_with_competitive_drug"] > 273 , 273 ,mendel["patients_treated_with_competitive_drug"])
mendel["patients_treated_with_competitive_drug"].describe()
mendel["patients_treated_with_competitive_drug"] = mendel["patients_treated_with_competitive_drug"].ffill().bfill()
mendel.isnull().sum()

#patients_treated_with_competitive_drug
#outlier treatment & handeling missing values

sns.boxplot(mendel['patients_treated_with_selling_drug']) 
centile(mendel,'patients_treated_with_selling_drug') 
mendel["patients_treated_with_selling_drug"] = np.where(mendel["patients_treated_with_selling_drug"] > 86 , 86 , mendel["patients_treated_with_selling_drug"])
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


#mendel.to_csv("mendel_final_dataset.csv", index = False)

mendel['pal'] = mendel['pal'].map(lambda x: x.lstrip('(-').rstrip('aAbBcC'))




#--------------------------------------------------------------------------------------------------
##percentage of sales accros categor
#    
sns.lmplot(y='actual_sales', x='percentage_territory_actual_sales', data=mendel)

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
#
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
#imp.fit(mendel['account_relation'])
#mendel['account_relation'] = imp.transform(mendel['account_relation'])
#
