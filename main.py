import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler
from sklearn.model_selection import cross_val_score, KFold
from sklearn.metrics import make_scorer, accuracy_score, precision_score, recall_score, f1_score


# Doc du lieu
df = pd.read_csv("Crime_Data_from_2020_to_Present.csv", sep=",")

# Lay thuoc tinh va nhan
data = df[['TIME OCC', 'AREA', 'Vict Age', 'Vict Sex', 'Vict Descent', 'Premis Cd', 'Weapon Used Cd', 'Crm Cd']]

data = data.dropna()
# print(data.info())

# Ma hoa nhan
le = LabelEncoder()
data['Vict Sex'] = le.fit_transform(data['Vict Sex'])
data['Vict Descent'] = le.fit_transform(data['Vict Descent'])
data['AREA'] = le.fit_transform(data['AREA'])


# Chuan hoa thoi gian
def categorize_time_period(time_occ):
    if 600 <= time_occ < 1200: # 06:00 to 12:00 Moring
        return 0
    elif 1200 <= time_occ < 1800 :  # 12:00 to 18:00 Afternoon
        return 1
    elif 1800 <= time_occ <= 2400 : # 18:00 to 24:00 Night
        return 2
    else: # Middal night
        return 3

def categorize_age(age):
    if 0 <= age <= 5:
        return 0
    elif 6 <= age <= 12:
        return 1
    elif 13 <= age <= 17:
        return 2
    elif 18 <= age <= 25:
        return 3
    elif 26 <= age <= 45:
        return 4
    elif 46 <= age <= 65:
        return 5
    else:
        return 6

def categorize_crime_type(crime_code):
    # Danh sách mã tội phạm cho từng loại
    violent_crimes = [
        110, 113, 121, 122, 210, 220, 230, 231,
        235, 236, 250, 251, 310, 320, 330, 622,
        623, 624, 625, 626, 627, 648, 753, 755,
        756, 810, 812, 813, 860, 865, 870, 880,
        882, 884, 886, 910, 921, 922, 928, 930,
        940, 950, 943, 944, 946, 948, 949
    ]

    sexual_offenses = [
        121, 122, 760, 761, 762, 763, 805, 806,
        810, 812, 813, 814, 815, 820, 821, 830,
        840, 845, 850
    ]

    theft_property_crimes = [
        210, 220, 310, 320, 330, 331, 341, 343,
        345, 347, 349, 350, 351, 352, 353, 354,
        410, 420, 421, 440, 441, 442, 443, 444,
        450, 451, 452, 473, 474, 480, 485, 487,
        510, 520, 522, 668, 670, 740, 745
    ]

    economic_fraud_crimes = [
        649, 651, 652, 653, 654, 660, 661,
        662, 664, 666, 668, 670, 950, 951, 956
    ]

    # Phân loại mã tội phạm
    if crime_code in violent_crimes:
        return 0  # Tội phạm nghiêm trọng liên quan đến tính mạng và bạo lực
    elif crime_code in sexual_offenses:
        return 1  # Tội phạm tình dục
    elif crime_code in theft_property_crimes:
        return 2  # Tội liên quan đến trộm cắp và tài sản
    elif crime_code in economic_fraud_crimes:
        return 3  # Tội phạm kinh tế và gian lận
    else:
        return 4  # Tội phạm liên quan đến hành vi xã hội và luật pháp


def categorize_location(premis_code):
    residential = {502, 501, 504, 505, 507, 510, 511, 514, 515, 516, 518, 519, 508, 509, 513}
    transportation = {101, 128, 124, 212, 801, 802, 804, 111, 113, 115, 122, 905, 910, 912, 890, 929, 937, 940, 950,
                      893, 745, 110}
    commercial = {405, 248, 404, 403, 410, 406, 412, 413, 201, 202, 210, 207, 233, 235, 244, 401, 402, 217, 237, 250}
    public_space = {102, 108, 109, 104, 107, 127, 141, 143, 144, 145, 146, 147, 109, 208, 209, 149, 243, 718, 756, 757}
    government_facility = {725, 726, 753, 214, 240}
    educational = {720, 721, 704, 722, 730, 731, 912}

    if premis_code in residential:
        return 0  # Khu dân cư
    elif premis_code in transportation:
        return 1  # Giao thông vận tải
    elif premis_code in commercial:
        return 2  #Thương mại
    elif premis_code in public_space:
        return 3  # Khu vực công cộng
    elif premis_code in government_facility:
        return 4  # Cơ sở chính phủ
    elif premis_code in educational:
        return 5  # Cơ sở giáo dục
    else:
        return 6  # Khu vực khác


data['TIME OCC'] = data['TIME OCC'].apply(categorize_time_period)
data['Crm Cd'] = data['Crm Cd'].apply(categorize_crime_type)
data['Premis Cd'] = data['Premis Cd'].apply(categorize_location)
data['Vict Age'] = data['Vict Age'].apply(categorize_age)


#Scale du lieu (chuan hoa du lieu dung min max)
scaler = MinMaxScaler()
data[['Premis Cd', 'Vict Age', 'Vict Sex', 'Vict Descent', 'AREA', 'TIME OCC', 'Weapon Used Cd']] = scaler.fit_transform(data[['Premis Cd', 'Vict Age','Vict Sex', 'Vict Descent', 'AREA', 'TIME OCC', 'Weapon Used Cd']])


X = data.drop(['Crm Cd'], axis = 1)
y = data['Crm Cd']


# Oversampling
from imblearn.over_sampling import RandomOverSampler
ros = RandomOverSampler()
X_res, y_res = ros.fit_resample(X, y)


# Chuyen du lieu thanh dang array
X_main = X_res.values
y_main = y_res.values

# Phan chia tap du lieu
X_train, X_test, y_train, y_test = train_test_split(X_main, y_main, test_size=0.2, random_state=42)