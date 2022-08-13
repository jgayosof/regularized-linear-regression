# imports:
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
from sklearn.linear_model import Lasso, LassoCV
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score


# Import dataset:
df_raw = pd.read_csv('https://raw.githubusercontent.com/4GeeksAcademy/regularized-linear-regression-project-tutorial/main/dataset.csv')


# Excercice: choose a variable to study:
# I chose 'Heart disease_prevalence'
df_heart_disease = pd.DataFrame(df_raw.corrwith(df_raw['Heart disease_prevalence'], axis=0), columns=['Correlation'])

# Which variables have a correlation grater than X=0.8:
X=0.8
corr_greather_X = df_heart_disease[abs(df_heart_disease['Correlation']) > X]
print(f'Will remove features with correlation > {X}: \n {corr_greather_X}')


# df_interim:
df_raw.to_csv('../data/raw/dataset_raw.csv')
df_interim = df_raw.copy()

# also remove 'COUNTY_NAME':
not_interest_variables = ['Heart disease_prevalence', 'Heart disease_Lower 95% CI', 'Heart disease_Upper 95% CI', 'COPD_prevalence', 'COPD_Lower 95% CI', 'COPD_Upper 95% CI', 'diabetes_prevalence', 'diabetes_Lower 95% CI', 'diabetes_Upper 95% CI', 'CKD_prevalence', 'CKD_Lower 95% CI', 'CKD_Upper 95% CI', 'COUNTY_NAME']
df_interim = df_raw.drop(not_interest_variables, axis=1)

df_interim.to_csv('../data/interim/dataset_interim.csv')

X = df_interim.copy()
y = df_raw['Heart disease_prevalence']

# Transform categorical variables to num (dummies)
X = pd.get_dummies(X, drop_first=True)
# train-test split:
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=13)


# Lasso Regression model with alpha = 0,3:
model_Lasso = Lasso(alpha=0.3, normalize=True)
model_Lasso.fit(X_train, y_train)

# Scale data!

# Lasso Regression (creation and trainning) with cross-validation to find optimal alpha
# ==============================================================================
# By default, LassoCV uses mean squared error
model_LassoCV = LassoCV(alphas=np.logspace(-10, 3, 200), normalize=True, cv=10)
aux = model_LassoCV.fit(X = X_train, y = y_train)


# Coefficient evolution as function of alpha
# ==============================================================================
alphas = model_LassoCV.alphas_
coefs = []

for alpha in alphas:
    modelo_temp = Lasso(alpha=alpha, fit_intercept=False, normalize=True)
    modelo_temp.fit(X_train, y_train)
    coefs.append(modelo_temp.coef_.flatten())

fig, ax = plt.subplots(figsize=(7, 3.84))
ax.plot(alphas, coefs)
ax.set_xscale('log')
ax.set_ylim([-15,None])
ax.set_xlabel('alpha')
ax.set_ylabel('coefficients')
ax.set_title('Coefficients as function of alpha')


# Error evolution as function of alpha 
# ==============================================================================
# model.mse_path_ stores the cv's mse for every alpha. Dimensions :  (n_alphas, n_folds)

mse_cv = model_LassoCV.mse_path_.mean(axis=1)
mse_sd = model_LassoCV.mse_path_.std(axis=1)

# square root to transform mse to rmse
rmse_cv = np.sqrt(mse_cv)
rmse_sd = np.sqrt(mse_sd)

# optimum and optimum + lstd
min_rmse = np.min(rmse_cv)
sd_min_rmse = rmse_sd[np.argmin(rmse_cv)]
optimo = model_LassoCV.alphas_[np.argmin(rmse_cv)]

# error +- stdv graph
fig, ax = plt.subplots(figsize=(7, 3.84))
ax.plot(model_LassoCV.alphas_, rmse_cv)
ax.fill_between(model_LassoCV.alphas_, rmse_cv + rmse_sd, rmse_cv - rmse_sd, alpha=0.2)

ax.axvline(x = optimo, c = "gray", linestyle = '--', label = 'optimum')

ax.set_xscale('log')
ax.set_ylim([0,None])
ax.set_title('cv error evolution as function of alpha')
ax.set_xlabel('alpha')
ax.set_ylabel('RMSE')
plt.legend()


# optimum alpha
opt_alpha = model_LassoCV.alpha_
print(f'Optimum alpha: {opt_alpha}')

model_Lasso_opt = Lasso(alpha=opt_alpha, normalize=True)
model_Lasso_opt.fit(X_train, y_train)


#save model:
joblib.dump(model_Lasso_opt, '../models/Lasso_health.pkl')