import tkinter as tk
from tensorflow.keras.models import load_model
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Load model and data
model = load_model('../models/wine_quality_model.h5')
data_red = pd.read_csv('../data/winequality-red.csv', delimiter=';')
data_white = pd.read_csv('../data/winequality-white.csv', delimiter=';')

def predict_quality():
    try:
        # Read input parameters
        params = [float(entry.get()) for entry in entries]
        
        # Predict quality
        quality = model.predict(np.array([params]))[0][0]
        quality_label.config(text=f"Predicted Quality: {quality:.2f}")

        # Determine wine type
        wine_type = "red" if params[0] > data_white['fixed acidity'].mean() else "white"
        plot_quality(quality, wine_type)
    except ValueError:
        tk.messagebox.showerror("Error", "Please enter valid numeric values.")

def plot_quality(pred_quality, wine_type):
    plt.figure(figsize=(8, 6))
    plt.scatter(data_red['alcohol'], data_red['quality'], color='red', alpha=0.5, label='Red Wines')
    plt.scatter(data_white['alcohol'], data_white['quality'], color='yellow', alpha=0.5, label='White Wines')
    color = 'red' if wine_type == 'red' else 'yellow'
    plt.scatter(params[10], pred_quality, edgecolors='blue', facecolors=color, s=200, linewidth=2, label='Predicted Wine Quality')
    plt.xlabel('Alcohol')
    plt.ylabel('Quality')
    plt.title('Wine Quality Prediction Visualization')
    plt.legend()
    plt.grid(True)
    plt.show()

# Create GUI
root = tk.Tk()
root.title("Wine Quality Prediction")
root.geometry("400x700")

entries = []
for i, label_text in enumerate(["Fixed Acidity", "Volatile Acidity", "Citric Acid", "Residual Sugar",
                                 "Chlorides", "Free Sulfur Dioxide", "Total Sulfur Dioxide", "Density", 
                                 "pH", "Sulphates", "Alcohol"]):
    tk.Label(root, text=f"{label_text}:").pack()
    entry = tk.Entry(root)
    entry.pack()
    entries.append(entry)

predict_button = tk.Button(root, text="Predict Quality", command=predict_quality)
predict_button.pack()

quality_label = tk.Label(root, text="Predicted Quality: ", font=("Helvetica", 14))
quality_label.pack()

root.mainloop()

