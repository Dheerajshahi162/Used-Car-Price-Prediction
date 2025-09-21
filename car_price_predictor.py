import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score

# 1. डेटा को लोड करें
file_path = 'C:\\Users\\WIN-11\\Documents\\csv file\\used car\\used_cars.csv'
df = pd.read_csv(file_path)

# 2. डेटा को साफ़ करें
df['price'] = df['price'].str.replace(r'[$,]', '', regex=True).astype(float)
df['milage'] = df['milage'].str.replace(r'[^\d]', '', regex=True).astype(int)
df['fuel_type'].fillna(df['fuel_type'].mode()[0], inplace=True)
df['accident'].fillna('Not Reported', inplace=True)
df['clean_title'].fillna('Not Reported', inplace=True)

# 3. आउटलायर हटाएँ
df = df[(df['price'] > 5000) & (df['price'] < 250000)].copy()

# 4. फ़ीचर्स (X) और टारगेट (y) को अलग करें
y = df['price']
X = df.drop('price', axis=1)

# 5. वन-हॉट एन्कोडिंग करें
X = pd.get_dummies(X, drop_first=True)

# 6. डेटा को ट्रेनिंग और टेस्टिंग सेट में विभाजित करें
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 7. मॉडल को प्रशिक्षित करें (Gradient Boosting Regressor)
gbr_model = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42)
gbr_model.fit(X_train, y_train)

# 8. मॉडल के प्रदर्शन का मूल्यांकन करें
y_pred_gbr = gbr_model.predict(X_test)
r2_gbr = r2_score(y_test, y_pred_gbr)
print("Final R-squared after removing outliers:", r2_gbr)

# 9. नई कार की कीमत का अनुमान लगाने के लिए डेटा तैयार करें
# हमें मॉडल को वही कॉलम देना होगा जो ट्रेनिंग डेटा में थे।
# हम एक खाली DataFrame बनाते हैं और फिर उसमें मान डालते हैं।
new_car_df = pd.DataFrame(columns=X.columns)

# नई कार के लिए एक लाइन का डेटा डालें
new_car_data = {
    'model_year': 2020,
    'milage': 35000,
    # यहाँ अन्य कैटेगोरिकल फ़ीचर जोड़ें, उदाहरण के लिए एक ब्रांड
    'brand_Audi': 1,
}

# नई कार के डेटा को DataFrame में भरें
for col in new_car_data:
    new_car_df.loc[0, col] = new_car_data[col]

# खाली कॉलमों को 0 से भरें
new_car_df.fillna(0, inplace=True)

# 10. अनुमानित कीमत प्रिंट करें
predicted_price = gbr_model.predict(new_car_df)
print(f"The estimated price for the new car is: ${predicted_price[0]:,.2f}")