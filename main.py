import numpy as np
import pandas as pd
from sklearn import neighbors, datasets
from ucimlrepo import fetch_ucirepo
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer


# Doc du lieu
df = pd.read_csv("Crime_Data_from_2020_to_Present.csv", sep=",")

# Lay thuoc tinh va nhan
data = df[['TIME OCC', 'AREA', 'Vict Age', 'Vict Sex', 'Vict Descent', 'Premis Cd', 'Weapon Used Cd', 'LOCATION', 'Crm Cd']]

# print(data.info())

# Kiem tra nhung cot co chua gia tri null
# for col in data.columns:
#     missing_data = data[col].isna().sum()
#     missing_precent = missing_data/len(data) * 100
#     print(f"Column {col}: has {missing_precent}% missing data")

# Chuyen du lieu thanh dang array
X = data.iloc[:,: -1 ].values
y = data.iloc[:, -1].values

# Tien hanh xu ly nhung thuoc tinh chua gia tri null
imputer = SimpleImputer(missing_values=np.nan, strategy="most_frequent")
imputer.fit(X[:, 3: 7])
X[:,3: 7] = imputer.transform(X[:,3 :7])

# Ma hoa du lieu bang LabelEncoder
encoder = LabelEncoder()
X[:, 3] = encoder.fit_transform(X[:, 3])
X[:, 4] = encoder.fit_transform(X[:, 4])
X[:, 7] = encoder.fit_transform(X[:, 7])

# Phan chia tap du lieu
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)