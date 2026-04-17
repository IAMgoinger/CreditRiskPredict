import pandas as pd
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
import warnings

warnings.filterwarnings("ignore")

csv_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "credit_risk_dataset.csv")

try:
    from data_EDA.raw import raw_data
except ImportError:
    try:
        from raw import raw_data
    except ImportError:
        raw_data = pd.read_csv(csv_path)

data = raw_data.copy()
data.dropna(axis=0, inplace=True)

if 'index' in data.columns:
    data.drop(['index'], axis=1, inplace=True)
data.reset_index(drop=True, inplace=True)

# Фильтрация выбросов
data = data[data['person_age'] <= 80]
data = data[data['person_emp_length'] <= 60]

# Бининг признаков
data['age_group'] = pd.cut(data['person_age'], bins=[20, 26, 36, 46, 56, 66],
                           labels=['20-25', '26-35', '36-45', '46-55', '56-65'])
data['income_group'] = pd.cut(data['person_income'], bins=[0, 25000, 50000, 75000, 100000, float('inf')],
                              labels=['low', 'low-middle', 'middle', 'high-middle', 'high'])
data['loan_amount_group'] = pd.cut(data['loan_amnt'], bins=[0, 5000, 10000, 15000, float('inf')],
                                   labels=['small', 'medium', 'large', 'very large'])

# Создание новых признаков
data['loan_to_income_ratio'] = data['loan_amnt'] / data['person_income']
data['loan_to_emp_length_ratio'] = data['person_emp_length'] / data['loan_amnt']
data['int_rate_to_loan_amt_ratio'] = data['loan_int_rate'] / data['loan_amnt']

X = data.drop(['loan_status'], axis=1)
y = data['loan_status']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=12)

# One-Hot Encoding
ohe_columns = ['cb_person_default_on_file', 'loan_grade', 'person_home_ownership',
               'loan_intent', 'income_group', 'age_group', 'loan_amount_group']

ohe = OneHotEncoder(sparse_output=False)
ohe.fit(X_train[ohe_columns])

ohe_train = ohe.transform(X_train[ohe_columns])
ohe_test = ohe.transform(X_test[ohe_columns])
ohe_feature_names = ohe.get_feature_names_out(ohe_columns)

ohe_data_train = pd.DataFrame(ohe_train, columns=ohe_feature_names, index=X_train.index)
ohe_data_test = pd.DataFrame(ohe_test, columns=ohe_feature_names, index=X_test.index)

X_new = pd.concat([ohe_data_train, X_train], axis=1).drop(ohe_columns, axis=1)
X_new_test = pd.concat([ohe_data_test, X_test], axis=1).drop(ohe_columns, axis=1)

# Standard Scaling
scale_cols = ['person_income', 'person_age', 'person_emp_length', 'loan_amnt', 'loan_int_rate',
              'cb_person_cred_hist_length', 'loan_percent_income', 'loan_to_income_ratio',
              'loan_to_emp_length_ratio', 'int_rate_to_loan_amt_ratio']

scaler = StandardScaler()

X_new[scale_cols] = X_new[scale_cols].astype(float)
X_new_test[scale_cols] = X_new_test[scale_cols].astype(float)

X_new.loc[:, scale_cols] = scaler.fit_transform(X_new.loc[:, scale_cols])
X_new_test.loc[:, scale_cols] = scaler.transform(X_new_test.loc[:, scale_cols])

print(f"Train set size: {X_new.shape}")
print(f"Test set size: {X_new_test.shape}")