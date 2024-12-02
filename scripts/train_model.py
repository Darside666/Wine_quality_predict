from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from data_preprocessing import load_and_preprocess_data
from sklearn.model_selection import train_test_split

# Load data
data = load_and_preprocess_data()
X = data.drop('quality', axis=1)
y = data['quality']

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define model
model = Sequential([
    Dense(64, input_dim=X_train.shape[1], activation='relu'),
    Dense(32, activation='relu'),
    Dense(1, activation='linear')
])

# Compile model
model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mae'])

# Train model
model.fit(X_train, y_train, epochs=100, batch_size=32, validation_split=0.2)

# Save model
model.save('../models/wine_quality_model.h5')
print("Model trained and saved to 'models/wine_quality_model.h5'.")
