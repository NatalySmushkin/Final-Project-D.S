# Final-Project-D.S
Telecom churn prediction
# Report

Most telecom companies suffer from customer churn. With a variety of service providers customer can easily switch from one operator to another. For companies it costs more to acquire a new customer than to retain an existing one. That is why telecom companies are interested in predicting customer churn.

#### Dataset

We have four datasets: 
* contract - the client's contract information (type of contract, date of commencement/termination of the contract,  payment method, paperless billing, monthly charges, total charges
* client — the client's personal data (gender, senior citizen or not, and if client has partners and dependents) 
* internet — information about Internet services that each client has signed up for (internet service, online security, online backup, device protection, tech support, streaming TV, streaming movies)
* phone — information about telephone services (if client has multiple lines or not)

In each dataset, the column 'customerID' contains a unique code assigned to each client.

The contract information is valid as of February 1, 2020.

First, we merged 4 datasets. Then, deleted missing values in column 'TotalCharges' and filled the rest missing values with "No service" assuming that client didn't have such service. Some columns were converted into appropriate types. 

Also, in this part we created two new columns: 'Tenure' that contains the information about how long client has been staying with company, and 'Churn' column that tells whether the customer has left ("1") or not ("0"). 'Churn' column is our target variable.

#### EDA 

Now, the data is ready for EDA.

Using plots we analyzed each variable separately and here are some conclusions:

* Most clients have month-to-month contract
* Equal number of men and women
* More clients have a paperless billing
* 6 times less senior citizens
* Equal distribution of individuals with partners and without
* About twice more individuals without dependents
* Most clients have internet through a fiber optic cable
* Most clients have electronic check
* Tenure Distribution (in month) shows us that many clients stay with us for short period of time (less than a year) and many clients stay for a long period (more than 5 years).
* We can see also that many clients have small charges (people usually choose the cheapest offer)

We also analyzed features in term of Churn and came to several conclusions:

* Those who have a month-to-month contract, fiber optic are more likely to churn
* Those who don't have such services like Online Scerurity, Online Backup, Device Protection, Tech Support are more likely to leave the company
* On average those who left the company stayed less than a year (10 months). Clients who do not churn, they tend to stay for a longer period (about 3 years)
* On average, clients who churn have higher monthly charges
* On average, total charges for those who churn are lower, because they stay for a short time


#### ML models

First, we prepeared data for ML models: 

* We converted categorical features with two options (Yes, No) and three options (Yes, No + No service -- the same meaning) into binary values Yes - 1, No - 0 (also, No and No service -- 0)

* For categorical features with more than two options (InternetService, Contract, PaymentMethod) we used pd.get_dummies() in order to convert them into numbers

* We scaled umeric features (Tenure, MonthlyCharges, TotalCharges) using StandardScaler()

* We devided data into train (80%) and test (20%) sets

After data was prepeared, we trained different models (Logistic Regression, Random Forest, XGBoost, CatBoost and LGBM) and tuned hyperparameters using RandomizedSearch.

Since we are dealing with imbalanced data, accuracy metric might be biased towards the majority class. We therefore choose ROC-AUC as our evaluation metric. 

We also looked at confusion metrics and can conclude that CatBoost had the smalest number of False Negatives. False Negatives mean that client will churn, but the model says he will not and we'll loose money. 


### After testing 5 models, we can conclude that CatBoost is a leader. CatBoost has the highest value of ROC_AUC (0.86).
