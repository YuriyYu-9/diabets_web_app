import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib
from tensorflow import keras

# Load the dataset
data = pd.read_csv('data/diabetes_1.csv')

# Split the data into features and target
X = data.drop(columns=['Outcome'])
y = data['Outcome']

# Standardize the data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Save the scaler for later use
joblib.dump(scaler, 'scaler.pkl')

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Build the model
model = Sequential()
model.add(Dense(16, input_dim=X_train.shape[1], activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(8, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(1, activation='sigmoid'))

# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(X_train, y_train, epochs=50, batch_size=16, validation_data=(X_test, y_test))

# Save the model
model.save('cnn_model.h5')


# Function to make predictions
def predict_diabetes(features):
    """
    Predicts whether a patient has diabetes based on the input features.

    :param features: List of patient features [pregnancies, glucose, blood_pressure, skin_thickness, insulin, bmi, diabetes_pedigree_function, age]
    :return: Probability of having diabetes
    """
    scaler = joblib.load('scaler.pkl')
    features = np.array(features).reshape(1, -1)
    features_scaled = scaler.transform(features)
    model = keras.models.load_model('cnn_model.h5')
    prediction = model.predict(features_scaled)
    return prediction[0][0]
