import streamlit as st
import pandas as pd
import numpy as np
from math import radians, sin, cos, sqrt, atan2
from sklearn.ensemble import RandomForestRegressor

# Load Dataset
df = pd.read_csv("uber.csv")

# Remove unwanted columns
df = df.drop(columns=['Unnamed: 0','key'], errors='ignore')
df = df.dropna()

# Haversine function
def haversine(lat1, lon1, lat2, lon2):

    R = 6371

    lat1, lon1, lat2, lon2 = map(radians,
                                 [lat1, lon1, lat2, lon2])

    dlat = lat2 - lat1
    dlon = lon2 - lon1

    a = sin(dlat/2)**2 + cos(lat1)*cos(lat2)*sin(dlon/2)**2
    c = 2 * atan2(sqrt(a), sqrt(1-a))

    return R * c

# Feature Engineering
df['distance_km'] = df.apply(lambda row:
    haversine(row['pickup_latitude'],
              row['pickup_longitude'],
              row['dropoff_latitude'],
              row['dropoff_longitude']), axis=1)

X = df[['pickup_latitude','pickup_longitude',
        'dropoff_latitude','dropoff_longitude',
        'passenger_count','distance_km']]

y = df['fare_amount']

# Train Model
model = RandomForestRegressor()
model.fit(X,y)

# Streamlit UI
st.title("Uber Fare Prediction App 🚖")

pickup_lat = st.number_input("Pickup Latitude")
pickup_lon = st.number_input("Pickup Longitude")

drop_lat = st.number_input("Dropoff Latitude")
drop_lon = st.number_input("Dropoff Longitude")

passengers = st.number_input("Passenger Count", min_value=1, max_value=6)

if st.button("Predict Fare"):

    distance = haversine(pickup_lat,pickup_lon,drop_lat,drop_lon)

    input_data = np.array([[pickup_lat,pickup_lon,
                            drop_lat,drop_lon,
                            passengers,distance]])

    prediction = model.predict(input_data)

    st.success(f"Estimated Fare: ${prediction[0]:.2f}")
