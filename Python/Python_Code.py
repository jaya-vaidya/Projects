
# coding: utf-8

# ### <center><b>Data Science with Python</b></center>
# ###  <center><b>USA Statistics</b></center>

# #### <center>Group 3 - Team Members<br><br>Jayalakshmi Vaidyanathan<br>Krutika Ambavane<br>Neha Narayankar</center>

# #### Importing all the required libraries:

# In[1]:

import pandas as pd
import numpy as np
import seaborn as sns
get_ipython().magic('pylab inline')
import sklearn as sk
import sklearn.tree as tree
from IPython.display import Image  
import pydotplus
import plotly.plotly as py
from plotly.graph_objs import *
pd.set_option('display.max_columns', None)
pd.set_option('precision', 2)
from plotly.offline import init_notebook_mode, iplot, plot
import plotly.graph_objs as go
init_notebook_mode(connected=True)
import plotly.tools as tls
import plotly


# In[2]:

# print all the outputs in a cell
from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity = "all"
import warnings
warnings.filterwarnings("ignore")


# ## Data set Description

# <li><b>df - USA Real Estate statistic dataset that contains information on Mortgage-Backed Securities,Geographic Business Investment,Real Estate Analysis </li>
#  <li>  <b> df2 - Census information dataset for each census tract in the USA </li>
#  <li>  <b> df_cen - Census information dataset aggregated for each state in USA </li>
#  <li>  <b> df3 - USA Real Estate statistic dataset aggregated for each state in USA </li>
#  <li>  <b> df_all - Merged dataset of of USA Census(df_cen) and USA Real Estate Statistics(df3)</li>

# The columns in <b> df - USA statistics dataset</b> are:
# <ol>
#     <li> <b>State :</b> State reported by the U.S. Census Bureau  </li>
# <li> state_ab : State Abbrevation </li>
# <li> city : City Name </li>
# <li> place : The place name reported by the U.S. Census Bureau for the specified geographic location. </li>
#     <li> <b>type :</b> The place Type reported by the U.S. Census Bureau for the specified geographic location </li>
# <li> primary : Defines whether the location is a tract location or a block group. </li>
# <li> zip_code : The closest zip code reported by the U.S. Education Department by the closest school relative to the Census Location </li>
# <li> area_code : The area code reported by the U.S. Census Bureau of the closest geographic location with area code information </li>
#     <li> <b>lat :</b> The latitude of geographic location </li>
# <li> <b>lng : </b>The longitude of geographic location </li>
# <li> ALand : The Square area of land at the geographic or track location </li>
# <li> AWater : The Square area of water at the geographic or track location </li>
# <li> <b>pop :</b> Male and female population of geographic location </li>
# <li> <b>male_pop :</b> Male population of geographic location </li>
# <li> <b>female_pop :</b> female population of geographic location </li>
# <li> <b>rent_mean :</b> The mean gross rent of the specified geographic location </li>
# <li> rent_median : The mean gross rent of the specified geographic location </li>
# <li> rent_stdev : The standard deviation of the gross rent for the specified geographic location </li>
# <li> rent_sample_weight : The sum of gross rent weight used in calculations </li>
# <li> rent_samples : The number of gross rent records used in the statistical calculations </li>
# <li> rent_gt_10 : The empirical distribution value that an individual’s rent will be greater than 10% of their household income in the past 12 months </li>
# <li> rent_gt_15 : The empirical distribution value that an individual’s rent will be greater than 15% of their household income in the past 12 months </li>
# <li> rent_gt_20 : The empirical distribution value that an individual’s rent will be greater than 20% of their household income in the past 12 months </li>
# <li> rent_gt_25 : The empirical distribution value that an individual’s rent will be greater than 25% of their household income in the past 12 months </li>
# <li> rent_gt_30 : The empirical distribution value that an individual’s rent will be greater than 30% of their household income in the past 12 months </li>
# <li> rent_gt_35 : The empirical distribution value that an individual’s rent will be greater than 35% of their household income in the past 12 months </li>
# <li> rent_gt_40 : The empirical distribution value that an individual’s rent will be greater than 40% of their household income in the past 12 months </li>
# <li> rent_gt_50 : The empirical distribution value that an individual’s rent will be greater than 50% of their household income in the past 12 months </li>
# <li> universe_samples : The size of the renter-occupied housing units sampled universe for the calculations </li>
# <li> used_samples : The number of samples used in the household income by gross rent as percentage of income in the past 12 months calculation </li>
# <li> <b>hi_mean : </b>The mean household income of the specified geographic location </li>
# <li> hi_median : The median household income of the specified geographic location </li>
# <li> hi_stdev : The standard deviation of the household income for the specified geographic location. </li>
# <li> hi_sample_weight : The number of households weighted used in the statistical calculations </li>
# <li> hi_samples : The number of households used in the statistical calculations </li>
# <li> family_mean : The mean family income of the specified geographic location </li>
# <li> family_median : The median family income of the specified geographic location </li>
# <li> family_stdev : he standard deviation of the family income for the specified geographic location. </li>
# <li> family_sample_weight : The number of family income weighted used in the statistical calculations </li>
# <li> family_samples : The number of family income used in the statistical calculations </li>
# <li> hc_mortgage_mean : The mean Monthly Mortgage and Owner Costs of specified geographic location </li>
# <li> hc_mortgage_median : The median Monthly Mortgage and Owner Costs of the specified geographic location. </li>
# <li> hc_mortgage_stdev : The standard deviation of the Monthly Mortgage and Owner Costs for a specified geographic location. </li>
# <li> hc_mortgage_sample_weight : The number of samples used in the statistical calculations </li>
# <li> hc_mortgage_samples : The number of samples used in the statistical calculations </li>
# <li> hc_mean : The mean Monthly Owner Costs of specified geographic location </li>
# <li> hc_median : The median Monthly Owner Costs of a specified geographic location </li>
# <li> hc_stdev : The standard deviation of the Monthly Owner Costs of a specified geographic </li>
# <li> hc_samples : The samples used in the calculation of the Monthly Owner Costs statistics </li>
# <li> hc_sample_weight : The samples used in the calculation of the Monthly Owner Costs statistics </li>
# <li> home_equity_second_mortgage : Percentage of homes with a second mortgage and home equity loan </li>
# <li> second_mortgage : percent of houses with a second mortgage </li>
# <li> home_equity : Percentage of homes with a home equity loan. </li>
# <li> <b>debt :</b> Percentage of homes with some type of debt </li>
# <li> second_mortgage_cdf : Cumulative distribution value of one minus the percentage of homes with a second mortgage. The value is used as a performance feature </li>
# <li> home_equity_cdf : Cumulative distribution value of one minus the percentage of homes with a home equity loan. The value is used as a performance feature </li>
# <li> debt_cdf : Cumulative distribution value of one minus the percentage of homes with any home related debt. The value is used as a performance feature. </li>
# <li> <b>hs_degree :</b> Percentage of people with at least high school degree </li>
# <li> hs_degree_male : Percentage of males with at least high school degree </li>
# <li> hs_degree_female : Percentage of females with at least high school degree </li>
# <li> male_age_mean : The mean male age of specified geographic location </li>
# <li> male_age_median : The median male age of specified geographic location </li>
# <li> male_age_stdev : The standard male age of specified geographic location </li>
# <li> male_age_sample_weight : The samples used in the calculation of the male age of specified geographic location </li>
# <li> male_age_samples : The samples used in the calculation of the male age of specified geographic location </li>
# <li> female_age_mean : The mean female age of specified geographic location </li>
# <li> female_age_median : The median female age of specified geographic location </li>
# <li> female_age_stdev : The standard female age of specified geographic location </li>
# <li> female_age_sample_weight : The samples used in the calculation of the female age of specified geographic location </li>
# <li> female_age_samples : The samples used in the calculation of the female age of specified geographic location </li>
# <li> pct_own : Percentage of Owners </li>
# <li> married : Percentage of people married </li>
# <li> married_snp : Percentage of people married snp </li>
# <li> separated : Percentage of people seperated </li>
# <li> divorced : Percentage of people divorced </li>
# 
# 
# </ol>
# 

# The columns in <b> df2 - USA Census Dataset </b> are:
# <ol>
# <li> State : State, DC, or Puerto Rico, String </li>
# <li> <b>County</b>: County or county equivalent , String </li>
# <li> TotalPop: Total population, Numeric </li>
# <li> Men: Number of men, Numeric </li>
# <li> Women: Number of women, Numeric </li>
# <li> Hispanic: % of population that is Hispanic/Latino, Numeric </li>
# <li> White: % of population that is white, Numeric </li>
# <li> Black: % of population that is black,  Numeric </li>
# <li> Native: % of population that is Native American or Native Alaskan, Numeric </li>
# <li> Asian: % of population that is Asian, Numeric </li>
# <li> Pacific: % of population that is Native Hawaiian or Pacific Islander, Numeric </li>
# <li> Citizen: Number of citizens, Numeric </li>
# <li> Income: Median household income (dollars), Numeric </li>
# <li> IncomeErr: Median household income error (dollars), Numeric </li>
# <li> <b>IncomePerCap</b>: Income per capita (dollars), Numeric </li>
# <li> IncomePerCapErr: Income per capita error (dollars), Numeric </li>
# <li> <b>Poverty</b>: % under poverty level, Numeric </li>
# <li> ChildPoverty: % of children under poverty level, Numeric </li>
# <li> Professional: % employed in management, business, science, and arts, Numeric </li>
# <li> <b>Service</b>: % employed in service jobs, Numeric </li>
# <li> <b>Office</b>: % employed in sales and office jobs, Numeric </li>
# <li> Construction: % employed in natural resources, construction, and maintenance, Numeric </li>
# <li> Production: % employed in production, transportation, and material movement, Numeric </li>
# <li> Drive: % commuting alone in a car, van, or truck, Numeric </li>
# <li> Carpool: % carpooling in a car, van, or truck, Numeric </li>
# <li> Transit: % commuting on public transportation, Numeric </li>
# <li> <b>Walk</b>: % walking to work, Numeric </li>
# <li> OtherTransp: % commuting via other means, Numeric </li>
# <li> WorkAtHome: % working at home, Numeric </li>
# <li> MeanCommute: Mean commute time (minutes), Numeric </li>
# <li> Employed: Number employed (16+), Numeric </li>
# <li> <b>PrivateWork</b>: % employed in private industry, Numeric </li>
# <li> <b>PublicWork</b>: % employed in public jobs, Numeric </li>
# <li> SelfEmployed: % self-employed, Numeric </li>
# <li> FamilyWork: % in unpaid family work, Numeric </li>
# <li> <b>Unemployment</b>: Unemployment rate (%), Numeric </li>
# 
# </ol>

# #### Reading the dataset and creating a dataframe for it:

# In[3]:

df = pd.read_csv("real_estate_db.csv", encoding = 'latin-1')


# In[4]:

print(df.shape)


# ### Cleaning the dataset:
# <b>Lets remove the unnecessary columns that we aren't going to analyze on.

# In[5]:

df_useful = df.drop([col for col in df if ('hc_' in col ) or ('sample_weight' in col) or ('ALand' in col) or
                    ('BLOCKID' in col) or ('cdf' in col) or ('stdev' in col) or ('median' in col) or
                    ('pct' in col) or ('gt_2' in col) or ('gt_4' in col) or ('gt_35' in col) or
                    ('gt_15' in col) or ('SUMLEVEL' in col) or ('_snp' in col) or (col.startswith('u'))], axis=1)


# <b>Handling the NaNs:

# In[6]:

df_useful.fillna(0, inplace = True)


# <b>Converting to lower case:

# In[7]:

df_useful.columns = [x.lower() for x in df_useful.columns]


# #### Removing outliers from required columns:

# In[8]:

h = df_useful["hi_mean"].quantile(0.99)


# In[9]:

df_useful = df_useful[df_useful["hi_mean"] < h]


# In[10]:

r = df_useful["rent_mean"].quantile(0.99)


# In[11]:

df_useful = df_useful[df_useful["rent_mean"] < r]


# In[12]:

m = df_useful["married"].quantile(0.99)


# In[13]:

df_useful = df_useful[df_useful["married"] < m]


# In[14]:

p = df_useful["pop"].quantile(0.99)


# In[15]:

df_useful = df_useful[df_useful["pop"] < p]


# In[17]:

print(df_useful.shape)


# #### Lets visualize an important feature - Type:

# In[18]:

percents = df_useful["type"].value_counts().round(2)

print("Type Count Values: ")
print(percents)

types = df_useful["type"].value_counts() / len(df_useful["type"]) * 100

labels = types.index.values.tolist()
values = types.tolist()

trace1 = go.Pie(labels=labels, values=values,hoverinfo='label+percent', marker=dict(line=dict(color='#000000', width=1)))

layout = go.Layout(title='Distribuition of Types', legend=dict(orientation="h"));

fig = go.Figure(data=[trace1], layout=layout)
iplot(fig)


# <div class="alert alert-success">
# <b>As we can see we have major rows of cities and towns so lets take a new dataset containing only town and city records
# </div>

# #### Creating dataset containing records belonging to cities and towns only:

# In[19]:

df_ct = df_useful[(df_useful.type == 'City') | (df_useful.type == 'Town')]


# #### Creating bins:

# In[20]:

df_bins = df_ct.copy()


# In[21]:

df_bins['married'] = pd.cut(df_bins.married, bins=4,include_lowest=True)


# In[22]:

df_bins['some_type_of_debt'] = pd.cut(df_bins.debt, bins=4,include_lowest=True)


# In[23]:

df_bins['high_school_degree'] = pd.cut(df_bins.hs_degree, bins=10,include_lowest=True)


# In[24]:

df_bins['rent'] = pd.cut(df_bins.rent_mean, bins=10,include_lowest=True)


# In[25]:

df_bins['household_income'] = pd.cut(df_bins.hi_mean, bins=[5000, 30000, 63000, 90000, 300000],include_lowest=True)


# In[26]:

df_bins['home_equity'] = pd.cut(df_bins.home_equity, bins=[0.0,0.2,0.4,0.6,0.8,1.0],include_lowest=True)


# #### Creating another dataset with only non-string values:

# In[27]:

df_no_strings = df_useful.iloc[:,9:]


# ### Analyzing the variations in the rent prices using world map:
# <b>According to our dataset , we have latitude and longitude for each record. So lets see the rent prices for each county.

# In[28]:

import plotly.figure_factory as ff
import numpy as np


# In[29]:

mapbox_access_token = 'pk.eyJ1Ijoia3J1dGlrYWFtYnZhbmUiLCJhIjoiY2poZmoxMjBjMTZ4aTM2bmduYnZtYXlrZCJ9._NLH_EGbJpqz3VR-rLv1mw'


# In[30]:

plotly.tools.set_credentials_file(username='krutika.a', api_key='iSsK12rHGuumSjzRhYDF')


# In[31]:

plot = df_useful.copy()
plot['State FIPS Code'] = plot['stateid'].apply(lambda x: str(x).zfill(2))
plot['County FIPS Code'] = plot['countyid'].apply(lambda x: str(x).zfill(3))
plot['FIPS'] = plot['State FIPS Code'] + plot['County FIPS Code']
plot.fillna(0, inplace=True)


# In[32]:

map_us = plot.groupby('FIPS')[['FIPS', 'rent_mean']].mean().reset_index()


# In[33]:

colorscale1 = ["#f7fbff","#ebf3fb","#deebf7","#d2e3f3","#c6dbef","#b3d2e9","#9ecae1",
              "#85bcdb","#6baed6","#57a0ce","#4292c6","#3082be","#2171b5","#1361a9",
              "#08519c","#0b4083","#08306b"]

colorscale = [
    'rgb(68.0, 1.0, 84.0)',
    'rgb(66.0, 64.0, 134.0)',
    'rgb(38.0, 130.0, 142.0)',
    'rgb(63.0, 188.0, 115.0)',
    'rgb(216.0, 226.0, 25.0)'
    'rgb(223.0, 223.0, 33.0)'
    'rgb(192.0, 226.0, 57.0)'
    'rgb(226.0,204.0,57.0)',
    'rgb(226.0,169.0,25.0)',
    'rgb(222.0,90.0,25.0)',
]
endpts = list(np.linspace(map_us['rent_mean'].min(), map_us['rent_mean'].max(), len(colorscale) - 1))
fips = map_us['FIPS'].tolist()
values = map_us['rent_mean'].tolist()


# In[34]:

fig = ff.create_choropleth(
    fips=fips, values=values,
    binning_endpoints=endpts,
    colorscale=colorscale,
    show_state_data=False,
    show_hover=True, centroid_marker={'opacity': 0},
    asp=2.9, title='Rent by county in United States ',
    county_outline={'color': 'rgb(255,255,255)', 'width': 0.5},
    legend_title='rent in United States',
)


# In[35]:

py.iplot(fig, filename='choropleth_full_usa')


# <div class="alert alert-success">
# <b>The above map shows the distribution of rent and helps us to visualize which locations have concentration of higher rent prices.<br>
# So we have found that specific counties in California and NewYork have the highest rent prices
# eg - Santa Clara County of California
#     <div>

# ## <b>Finding 1:</b> rent vs debt

# In[36]:

import matplotlib.pyplot as pyplt
sns.set_context("notebook", font_scale=1.3)


f, ax = pyplot.subplots(figsize=(18, 5))
fig = sns.pointplot(x='rent', y='debt', data=df_bins, ax=ax)
pyplt.xlabel('Ranges of Rental values')
pyplt.ylabel('percent housesholds with Debt')
pyplt.title('(Rent) vs (Percent households with debt)')
sns.despine() #to remove top and right line on graph
pyplt.xticks(rotation=30, ha='right')

pyplt.show(fig);


# <div class="alert alert-success">
# <b>In real world, we wont expect rent and debt to have any correlation. However, from the above graph we can interpret that locations with higher mean rent have higher percentage of houses with debt. 
# </div>

# In[37]:

df_bins.groupby('state')['state'].count().nlargest(10)


# <div class="alert alert-success">
# <b>As we can see, this dataset has largest records for the state of California. So lets study a little about this state.
# </div>

# ### Creating a dataset containing records belonging only to the state of California:

# In[38]:

df_cali = df_bins[(df_bins.state == 'California')]


# In[39]:

sns.set_context("notebook", font_scale=1.3)
f, ax = pyplot.subplots(figsize=(18, 5))
fig = sns.pointplot(x='rent', y='debt', data=df_cali, ci = None, ax=ax)
pyplt.xlabel('Ranges of Rental values')
pyplt.ylabel('Percent housesholds with Debt')
pyplt.title('CALIFORNIA \n (Rent) vs (Percent households with debt)')
sns.despine() #to remove top and right line on graph
pyplt.xticks(rotation=30, ha='right')
pyplt.show(fig);


# <div class="alert alert-success">
# <b>Here we just focused on one state - California. We observe similar correlation between mean rent and percentage of houses having debts.
# </div>

# <b>Lets search for top 5 states with largest mean rent and largest percentage of houses with debt.

# In[40]:

df_bins.groupby('state')['debt'].mean().nlargest(5) 


# In[41]:

df_bins.groupby('state')['rent_mean'].mean().nlargest(5)


# <div class="alert alert-success">
# <b>This also supports our conclusion that the states having highest rents also have highest percentages of household with debts.
# </div>

# <b>Now, lets examine the reason behind people having higher rents. Lets see if the type of the place people reside in, matters.

# In[42]:

df_bins.groupby('type')['pop'].sum()


# In[43]:

df_bins.groupby('type')['rent_mean'].mean() 


# In[44]:

sns.set_context("notebook", font_scale=1.2)
fig = sns.factorplot(x='type',y='rent_mean',data=df_bins,kind='bar', size=6, aspect=0.8)
pyplt.xlabel('Location Type')
pyplt.ylabel('Mean Rent')
pyplt.title('(Mean Rent) vs (Type of Location)')

sns.despine() 
pyplt.show(fig);


# ## Conclusion :

# 
# <div class="alert alert-success">
# <b>As we can see from above, the population is highest in cities resulting in an increase in demands, further leading to higher rent prices there compared to towns.
# </div>

# ***

# ## Finding 2:   rent vs income

# <h3> <b>Now lets explore the reason as to why these people from cities can afford higher rents. </h3>

# In[45]:

sns.set_context("notebook", font_scale=1.8)
fig = sns.lmplot(x='rent_mean', y='hi_mean', data=df_bins,    
           fit_reg=False,
           hue='type', aspect = 3)   
pyplt.xlabel('Mean Rental Rates')
pyplt.ylabel('Household Income level')
pyplt.title('Rental Rates vs Household Income')
sns.despine() #to remove top and right line on graph
pyplt.show(fig);


# In[46]:

sns.set_context("notebook", font_scale=1.5)
sns.factorplot(x='household_income',y='rent_mean',hue='type',data=df_bins,kind='bar',aspect = 3)
pyplt.xlabel('Household income')
pyplt.ylabel('Mean rental values')
pyplt.title('Mean Rent Rates vs Household Income')
pyplt.show(fig);


# <div class="alert alert-success">
# <b>From above, we can conclude that as the mean household income goes on increasing, the mean rent prices also go on increasing. We can also deduce that the people from city have more household income and more rent than the town people.
# <br>   
# <b>This above graph also confirms the same that higher income areas have higher rents.
# </div>

# #### Lets try to predict the rent prices for cities and towns using Linear Regression:
# We will use **Simple Regresssion** as we are going to predict only one variable.

# **Step 1.** Importing the required libraries:</div>

# In[47]:

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_predict
import matplotlib.pyplot as plt


# **Step 2.** Create X and Y: <br>
# >Our Y will be the rent prices that we are going to predict.

# In[48]:

X = df_no_strings.drop('rent_mean',axis = 1)
Y = df_no_strings.rent_mean


# **Step 3.** Split the data into train and test data: <br>
# >We have made our train data as 75% of the original data and test data as 25%. We have randomized the splitting by using random state and given labels to both our data.
# 

# In[49]:

train_features, test_features, train_labels, test_labels = train_test_split(X, Y, test_size = 0.25, random_state = 42)


# In[50]:

train_features.shape


# **Step 4.** Instantiate and fit our data the model:

# In[51]:

rf = LinearRegression()
rf.fit(train_features, train_labels);


# **Step 5.** Check score of our model on test data:

# In[52]:

rf.score(test_features, test_labels)


# **Step 6.** Calculate the error rates:

# In[53]:

predictions = rf.predict(test_features)
errors = abs(predictions - test_labels)


# In[54]:

print('Mean Absolute Error:', round(np.mean(errors), 2), 'dollars')


# In[55]:

print('Root Mean square error:', sqrt(round(np.mean(errors), 2)), 'dollars')


# **Step 5:** Plotting graph: <br>
# > Let's visualize the predicted values vs the measured values.

# In[56]:

predicted = cross_val_predict(rf, train_features, train_labels, cv=10)


# In[57]:

sns.set_context("notebook", font_scale=1.8)
y = train_labels
fig, ax = plt.subplots()
fig.set_size_inches(18.5, 10.5)
ax.scatter(y, predicted, edgecolors=(0, 0, 0))
ax.plot([y.min(), y.max()], [y.min(), y.max()], 'k--', lw=4)
ax.set_xlabel('Measured Rent Values')
ax.set_ylabel('Predicted Rent Values')
plt.title('Measured vs Predicted Rent values')
plt.show();


# <div class="alert alert-success">
# <b>The accuracy of the predicted rent is ```0.71.``` We can see the upward trend and the points aligned with the linear function line.
# </div>

# ## Conclusion:

# <div class="alert alert-success">
# Variables that **highly contribute** to **rent** : **[ household income, debt, home equity, highschool degree, second mortgage ] **
# <br><br>
# <hr/>
# <b>We can conclude this because using these factors, we trained the above model and it is able to predict rent with accuracy of ```71%.```
# <b>We can come to this conclusion by also looking at the heat map plotted below which shows us the variables with good correlation.
# </div>

# ### Finding the correlation and important variables:

# In[58]:

df_corr = df_no_strings.dropna()


# In[59]:

data_1 = ['rent_gt_10','rent_gt_30','rent_gt_50', 'rent_mean','zip_code','area_code']
data_2 = ['home_equity', 'debt', 'hs_degree','hi_mean' , 'second_mortgage']
data_3 = ['married','divorced','separated'] 


# create df_corr dataframe & drop all nans:
data = data_1 + data_2 + data_3
df_corr = df_no_strings[data].dropna()
df_corr[data] = df_corr[data].apply(lambda x: (x - np.mean(x))/np.std(x))
corrmat = df_corr.corr(method='spearman')


# Draw the heatmap using seaborn
f = plt.figure(figsize=(10, 10))
sns.set(font_scale= 1.5,rc={"font.size": 2.1})
mask = np.zeros_like(corrmat); mask[np.triu_indices_from(mask)] = True
with sns.axes_style("white"): ax = sns.heatmap(corrmat, mask=mask, vmax=0.5, square=True)
plt.title("variables correlation overview", fontsize=20); plt.show();


# ------------------------------------------------------------------------------------------------------------

# ## Finding 3:

# <b> In order to find out the trends based on State, Race and other census variables, we are merging the Real estate data with the USA Census Data

# ## Census Data

# <b> Data preparation of USA Census Dataset

# In[109]:

df2 = pd.read_csv('census.csv')


# In[110]:

df2.head()


# In[111]:

df2.drop(['CensusTract'], axis=1,inplace=True)


# In[112]:

df2 = df2.dropna()


# 
# 
# 
# <b> Aggregation: For aggregating the data for each state ,we need to convert the metrics expressed as percentage for each county into counts so they can be summed.

# In[113]:

#Race 
df2['count_hisp']=df2.TotalPop*df2.Hispanic/100
df2['count_white']=df2.TotalPop*df2.White/100
df2['count_black']=df2.TotalPop*df2.Black/100
df2['count_native']=df2.TotalPop*df2.Native/100
df2['count_asian']=df2.TotalPop*df2.Asian/100
df2['count_pacific']=df2.TotalPop*df2.Pacific/100

#Income/Work 
df2['total_income']=df2.TotalPop*df2.IncomePerCap
df2['count_childpoverty']=df2.TotalPop*df2.ChildPoverty/100
df2['count_poverty']=df2.TotalPop*df2.Poverty/100
df2['count_professional']=df2.TotalPop*df2.Professional/100
df2['count_service']=df2.TotalPop*df2.Service/100
df2['count_office']=df2.TotalPop*df2.Office/100
df2['count_construction']=df2.TotalPop*df2.Construction/100
df2['count_production']=df2.TotalPop*df2.Production/100

#Mode of commute
df2['count_drive']=df2.TotalPop*df2.Drive/100
df2['count_carpool']=df2.TotalPop*df2.Carpool/100
df2['count_transit']=df2.TotalPop*df2.Transit/100
df2['count_walk']=df2.TotalPop*df2.Walk/100
df2['count_othertransp']=df2.TotalPop*df2.OtherTransp/100
df2['count_workathome']=df2.TotalPop*df2.WorkAtHome/100
df2['count_meancommute']=df2.TotalPop*df2.MeanCommute

# Work 
df2['count_privatework']=df2.TotalPop*df2.PrivateWork/100
df2['count_publicwork']=df2.TotalPop*df2.PublicWork/100
df2['count_selfemployed']=df2.TotalPop*df2.SelfEmployed/100
df2['count_familywork']=df2.TotalPop*df2.FamilyWork/100
df2['count_unemployment']=df2.TotalPop*df2.Unemployment/100


# In[114]:

df2.head()


# In[115]:

df_cen = df2.copy()


# In[116]:

df_cen.drop(['Income','Hispanic', 'White', 'Black','Native','Asian', 'Pacific', 'IncomePerCap', 'Poverty', 
             'ChildPoverty', 'Professional', 'Service', 'Office', 'Construction', 'Production', 'Drive', 'Carpool', 
             'Transit', 'Walk', 'OtherTransp', 'WorkAtHome', 'MeanCommute', 'PrivateWork', 'PublicWork', 
             'SelfEmployed', 'FamilyWork', 'Unemployment', 'IncomeErr','IncomePerCapErr','Men','Women'], axis=1,inplace=True)


# In[117]:

df_cen.head(2)


# In[118]:

df_cen = df_cen.groupby('State').agg({'State': 'first','TotalPop':'sum','Citizen': 'sum','count_hisp': 'sum', 'count_white': 'sum', 
                                            'count_black': 'sum','count_native': 'sum','count_asian': 'sum', 
                                            'count_pacific': 'sum', 'total_income': 'sum', 'count_poverty': 'sum', 
                                            'count_childpoverty': 'sum', 'count_professional': 'sum', 
                                            'count_service': 'sum', 'count_office': 'sum', 'count_construction': 'sum',
                                            'count_production': 'sum', 'count_drive': 'sum', 'count_carpool': 'sum', 
                                            'count_transit': 'sum', 'count_walk': 'sum', 'count_othertransp': 'sum', 
                                            'count_workathome': 'sum', 'count_meancommute': 'sum', 'Employed': 'sum', 
                                            'count_privatework': 'sum', 'count_publicwork': 'sum', 
                                            'count_selfemployed': 'sum', 'count_familywork': 'sum', 
                                            'count_unemployment': 'sum'})


# In[119]:

df_cen.head()


# <b> Converting the counts to percent per state population

# In[120]:

#Demographics

df_cen['Hispanic']=df_cen.count_hisp/df_cen.TotalPop*100
df_cen['White']=df_cen.count_white/df_cen.TotalPop*100
df_cen['Black']=df_cen.count_black/df_cen.TotalPop*100
df_cen['Native']=df_cen.count_native/df_cen.TotalPop*100
df_cen['Asian']=df_cen.count_asian/df_cen.TotalPop*100
df_cen['Pacific']=df_cen.count_pacific/df_cen.TotalPop*100

#Income 
df_cen['IncomePerCap']=df_cen.total_income/df_cen.TotalPop
df_cen['Poverty']=df_cen.count_poverty/df_cen.TotalPop*100
df_cen['ChildPoverty']=df_cen.count_childpoverty/df_cen.TotalPop*100
df_cen['Professional']=df_cen.count_professional/df_cen.TotalPop*100
df_cen['Service']=df_cen.count_service/df_cen.TotalPop*100
df_cen['Office']=df_cen.count_office/df_cen.TotalPop*100
df_cen['Construction']=df_cen.count_construction/df_cen.TotalPop*100
df_cen['Production']=df_cen.count_production/df_cen.TotalPop*100

#Mode of commute
df_cen['Drive']=df_cen.count_drive/df_cen.TotalPop*100
df_cen['Carpool']=df_cen.count_carpool/df_cen.TotalPop*100
df_cen['Transit']=df_cen.count_transit/df_cen.TotalPop*100
df_cen['Walk']=df_cen.count_walk/df_cen.TotalPop*100
df_cen['Othertransp']=df_cen.count_othertransp/df_cen.TotalPop*100
df_cen['Workathome']=df_cen.count_workathome/df_cen.TotalPop*100
df_cen['Meancommute']=df_cen.count_meancommute/df_cen.TotalPop

# Work Type
df_cen['Privatework']=df_cen.count_privatework/df_cen.TotalPop*100
df_cen['Publicwork']=df_cen.count_publicwork/df_cen.TotalPop*100
df_cen['Selfemployed']=df_cen.count_selfemployed/df_cen.TotalPop*100
df_cen['Familywork']=df_cen.count_familywork/df_cen.TotalPop*100
df_cen['Unemployment']=df_cen.count_unemployment/df_cen.TotalPop*100
df_cen['Employed']=df_cen.Employed/df_cen.TotalPop*100


# In[121]:

df_cen.drop(['TotalPop'], axis=1,inplace=True)


# In[122]:

df_cen.head()


# In[123]:

df_cen.columns = [x.lower() for x in df_cen.columns]


# ## USA Real Estate Statistics Data

# <b> Data preparation of USA Real Estate statistics data so that they can be merged with USA Census data based on states

# In[124]:

df3 = pd.read_csv("real_estate_db.csv", encoding = 'latin-1')


# In[125]:

df3 = df3.drop([col for col in df3 if ('hc_' in col ) or ('sample_weight' in col) or ('ALand' in col) or
                    ('BLOCKID' in col) or ('cdf' in col) or ('stdev' in col) or ('median' in col) or
                    ('pct' in col) or ('gt_2' in col) or ('gt_4' in col) or ('gt_35' in col) or
                    ('gt_15' in col) or ('SUMLEVEL' in col) or ('_snp' in col) or (col.startswith('u') or ('primary' in col) or ('uid' in col)or
                                                                                  ('blockid' in col))], axis=1)


# In[126]:

df3.head()


# In[127]:

df3.fillna(0, inplace = True)


# In[128]:

# Work 
df3['total_pop'] =(df3.male_pop+df3.female_pop)
df3['count_second_mortgage']=(df3.total_pop*df3.second_mortgage)/100
df3['count_debt']=(df3.total_pop*df3.debt)/100
df3['count_hs_degree']=(df3.total_pop*df3.hs_degree)/100


# In[129]:

df3.drop(['pop','second_mortgage', 'debt', 'hs_degree'], axis=1,inplace=True)


# In[130]:

df3 = df3.groupby('state').agg({'state': 'first','total_pop': 'sum','female_age_mean': 'mean','male_age_mean': 'mean',
                                            'rent_mean': 'mean','hi_mean': 'mean','count_second_mortgage': 'sum', 
                                             'count_debt': 'sum', 'count_hs_degree': 'sum'
                                            })


# In[131]:

# Work 
df3['second_mortgage_pct']=df3.count_second_mortgage/df3.total_pop*100
df3['debt_pct']=df3.count_debt/df3.total_pop*100
df3['high_school_degree_pct']=df3.count_hs_degree/df3.total_pop*100


# In[132]:

df3.head()


# ## Merge : USA Census and USA Real Estate Statistics data

# <b> Merging the USA Census Data and USA Real Estate statistics based on State

# In[133]:

df_all = df3.merge(df_cen, left_on = 'state', right_on='state')


# In[134]:

df_all.head()


# ### Initial Analysis Plots

# <b>Variable correlation

# In[135]:


data_1 = ['high_school_degree_pct','rent_mean','employed']
data_2 = ['privatework','selfemployed','publicwork', 'familywork']
data_3 = ['black', 'white', 'hispanic', 'asian'] 
data_4 = ['professional','service','office','construction','production'] 


# create df_corr dataframe & drop all nans:
data = data_1 + data_2 + data_3+data_4
df_corr = df_all[data].dropna()
df_corr[data] = df_corr[data].apply(lambda x: (x - np.mean(x))/np.std(x))
corrmat = df_corr.corr(method='spearman')


# Draw the heatmap using seaborn
f = plt.figure(figsize=(10, 10))
sns.set(font_scale= 1.5,rc={"font.size": 2.1})
mask = np.zeros_like(corrmat); mask[np.triu_indices_from(mask)] = True
with sns.axes_style("white"): ax = sns.heatmap(corrmat, mask=mask, vmax=0.5, square=True)
plt.title("variables correlation overview", fontsize=20); plt.show();


# In[136]:

dfallmap = df_all.copy()

US_STATE_ABB = {
    'Alabama': 'AL',
    'Alaska': 'AK',
    'Arizona': 'AZ',
    'Arkansas': 'AR',
    'California': 'CA',
    'Colorado': 'CO',
    'Connecticut': 'CT',
    'Delaware': 'DE',
    'District of Columbia': 'DC',
    'Florida': 'FL',
    'Georgia': 'GA',
    'Hawaii': 'HI',
    'Idaho': 'ID',
    'Illinois': 'IL',
    'Indiana': 'IN',
    'Iowa': 'IA',
    'Kansas': 'KS',
    'Kentucky': 'KY',
    'Louisiana': 'LA',
    'Maine': 'ME',
    'Maryland': 'MD',
    'Massachusetts': 'MA',
    'Michigan': 'MI',
    'Minnesota': 'MN',
    'Mississippi': 'MS',
    'Missouri': 'MO',
    'Montana': 'MT',
    'Nebraska': 'NE',
    'Nevada': 'NV',
    'New Hampshire': 'NH',
    'New Jersey': 'NJ',
    'New Mexico': 'NM',
    'New York': 'NY',
    'North Carolina': 'NC',
    'North Dakota': 'ND',
    'Ohio': 'OH',
    'Oklahoma': 'OK',
    'Oregon': 'OR',
    'Pennsylvania': 'PA',
    'Rhode Island': 'RI',
    'Puerto Rico': 'PR',
    'South Carolina': 'SC',
    'South Dakota': 'SD',
    'Tennessee': 'TN',
    'Texas': 'TX',
    'Utah': 'UT',
    'Vermont': 'VT',
    'Virginia': 'VA',
    'Washington': 'WA',
    'West Virginia': 'WV',
    'Wisconsin': 'WI',
    'Wyoming': 'WY',
}

dfallmap['state'] = dfallmap['state'].map(US_STATE_ABB)


# <b>Below graph shows the distribution of employed among all states of USA. 

# In[137]:

sns.factorplot(y='employed', data=dfallmap, x='state',aspect=5,
               kind='point', order=US_STATE_ABB.values())


# <b>Lets now identify other metrics that affect the employed

# In[138]:

sns.distplot(dfallmap.privatework, bins=20)


# <b>The Private work sector is skewed towards left

# In[139]:

sns.distplot(dfallmap.high_school_degree_pct, bins=20)


# <b>The High school degree is skewed towards left

# <div class="alert alert-success">
# <b>From the above distplots, it is very clear that the private work and high school degree have similar distribution (skewed left). Now lets relate with other Census metrics and explore further.
#     </div>

# <b>Among the employed population, the work type is categorized as Private, Self Employed ,Public work and Family work
# 

# <b> Lets analyse their distribution in USA

# In[140]:

df_mean = df_all[['privatework', 'selfemployed','publicwork','familywork']].copy()


# In[141]:

workmean = df_mean.mean()
workmean


# In[142]:

workmean = workmean.to_frame()
workmean.columns = ['WorkRate']


# In[143]:

import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import matplotlib as mpl
import matplotlib.pyplot as plt

# Font size
mpl.rcParams['font.size'] = 14.0

# Showing the labels of the Pie-Chart
labels = 'Private Work', 'Public Work', 'Self Employed', 'Family Work' 
fracs = workmean.WorkRate.tolist()

the_grid = GridSpec(1,1)
plt.subplot(the_grid[0, 0], aspect=1)

patches, texts, autotexts = plt.pie(fracs, labels=labels , autopct='%1.1f%%', startangle=90, radius=1.5)
                                    

texts[0].set_fontsize(18)
texts[1].set_fontsize(18)
texts[2].set_fontsize(18)
texts[3].set_fontsize(18)
plt.tight_layout();


# <div class="alert alert-success">
# <b>From the above Pic Chart it is very clear that Private work is predominant among all States of USA
#     </div>

# ### Finding 3: Private Work Sector Trend Among Black Population

# <b>Lets further Dig into the Private work Sector

# In[144]:

df_Private = df_all.copy()


# In[145]:

df_Private.privatework.mean()


# <b>Creating Binary column private_high for high and low private work Sector based on its mean

# In[146]:

df_Private['private_high']=(df_Private.privatework > 77.79)*1.0


# In[147]:

df_Private[['state','private_high']].head()


# In[148]:

df_Private.drop(['state','total_pop','count_second_mortgage','count_debt','count_hs_degree'], axis=1,inplace=True)


# In[149]:

df_Private = df_Private.drop([col for col in df_Private if ('count_' in col ) or ('state' in col) or ('total_pop' in col) ], axis=1)


# In[150]:

df_Private.head(2)


# ### Finding - 3: Machine Learning

# In[151]:

X = df_Private.drop(['private_high','privatework','publicwork','selfemployed','familywork','unemployment','meancommute','poverty',
                     'childpoverty','total_income','citizen','construction'],axis=1)


# In[152]:

Y = df_Private.private_high


# In[153]:

dt = tree.DecisionTreeClassifier(max_depth=2)
dt.fit(X,Y)


# In[154]:

dt_feature_names = list(X.columns)
dt_target_names = [str(s) for s in Y.unique()]
tree.export_graphviz(dt, out_file='tree.dot', 
    feature_names=dt_feature_names, class_names=dt_target_names,
    filled=True)  
graph = pydotplus.graph_from_dot_file('tree.dot')
Image(graph.create_png())


# <div class="alert alert-success">
# <b>Within states whose black population is less than 3.6%, and where less than 16.9% have service job , work privately. State whose black population is greater than 3.6%  have higher service based private work rate and they pay rent of less than or equal to 1381.0
# </div>
# 

# ###  Finding-3 : Validation

# In[156]:

df_Private_val = df_all.copy()
df_Private_val['black_pop'] = pd.cut(df_Private_val['black'],bins=[0,4,20,100])


# In[157]:

df_Private_val.service.mean()


# <b> Creating binary Column service_high for high and low Service work based on mean value

# In[158]:

df_Private_val['service_high']=(df_Private_val.service > 18.48)*1.0


# In[159]:

df_Private_val[['state','service_high']].head()


# In[160]:

g =sns.factorplot(x ='black_pop', y='privatework', hue='service_high', 
                  data=df_Private_val, aspect=3, kind='bar', 
                  legend_out = True)
# replace labels
g.set_axis_labels("Black pop %", "Private work Rate")
g.set_titles("{col_name} {col_var}")
g.set(ylim=(0, 90))
g.despine(left=True) 
new_title = '% of Service Jobs'
g._legend.set_title(new_title)
new_labels = ['Low', 'High']
for t, l in zip(g._legend.texts, new_labels): t.set_text(l)


# <b> Bar Graph Showing the private work based service job distribution among the Black Population 

# <div class="alert alert-success">
# <b> From the above bar graph it is clear that rate of private work based service job are lower for states with Black population below 4% 
#     and for states with  population of Balck greater than 4%, have higher rate of private work based service job 
#     </div>

# <b>The below are the top five states with maximum private work based service job percentage among black population (greater than 3.6%)

# In[161]:

df_all[df_all.black > 3.6].sort_values(by=['privatework','black'],ascending=False)[['state','privatework','service','high_school_degree_pct','black','hi_mean','rent_mean',]].rename(columns={'state':'State','privatework':'Private_Work','service':'Service','black':'Black_Pop_Pct','high_school_degree_pct':'high_school_degree','hi_mean':'Income','rent_mean':'Rent'}).head(5)


# ### Conclusion

# <div class="alert alert-success">
# <b>Despite Private work type being predominant in USA, the top 5 state with highest rate of private work based service job have lower income among the Black population, even though their High school degree percentage of the state is high.They are in a position to afford only lower rent. For betterment of their lifes,these states should take more efforts in increasing the salary of private work based service jobs among the educated black population.
# </div>
# 
