import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import os
import json
import traceback

model = load_model('models/EfficientNetB1/model_224.keras')

with open('models/EfficientNetB1/class_indices.json', 'r') as f:
    class_indices = json.load(f)

class_labels = [class_name for class_name, idx in sorted(class_indices.items(), key=lambda item: item[1])]

def preprocess_image(image_path):
    img = tf.keras.preprocessing.image.load_img(image_path, target_size=(224, 224))
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = tf.keras.applications.mobilenet_v2.preprocess_input(img_array)  
    return img_array

def predict(image_path):
    try:
        img = preprocess_image(image_path)
        prediction = model.predict(img, verbose=0)

        print(f"Prediction vector: {prediction}")
        predicted_class_index = np.argmax(prediction)
        predicted_class = class_labels[predicted_class_index]
        confidence = float(np.max(prediction))
        actual_class = os.path.basename(os.path.dirname(image_path))

        return predicted_class, confidence, actual_class
    except Exception as e:
        print("Prediction Error:", e)
        traceback.print_exc()
        return "Error", 0.0, "Unknown"

def open_file():
    file_path = filedialog.askopenfilename()
    if file_path:
        img = Image.open(file_path).resize((300, 300))
        img_tk = ImageTk.PhotoImage(img)
        image_label.config(image=img_tk)
        image_label.image = img_tk

        result_label.config(text="Predicting...", bg="#fff0f5", fg="black", relief="flat", bd=0)
        root.update_idletasks()

        predicted_class, confidence, actual_class = predict(file_path)
        if predicted_class == "Error":
            result_label.config(
                text="❌ Failed to Predict. Check console for details.",
                fg="red", bg="#fff0f5", relief="flat", bd=0
            )
        else:
            result_label.config(
                text=f"✨ Predicted: {predicted_class}\n🎯 Confidence: {confidence:.2f}\n🌺 Actual: {actual_class}",
                fg="white", bg="#7b4fa3", relief="solid", bd=3
            )

root = tk.Tk()
root.title("🌸 Flower Classifier - EfficientNetB1 🌻")
root.configure(bg="#fff0f5")
root.geometry("620x720")

frame = tk.Frame(root, bg="#fff0f5")
frame.pack(pady=30)

btn = tk.Button(
    frame,
    text="🌼 Choose Flower Image",
    command=open_file,
    font=("Verdana", 14, "bold"),
    bg="#c8e6c9",
    fg="#2e7d32",
    padx=20,
    pady=12,
    relief="raised",
    bd=4
)
btn.pack(pady=20)

image_label = tk.Label(frame, bg="#fff0f5")
image_label.pack(pady=10)

result_label = tk.Label(
    frame,
    text="",
    font=("Georgia", 16),
    bg="#fff0f5",
    fg="white",
    wraplength=500,
    justify="center",
    padx=20,
    pady=20,
    relief="flat",
    bd=0
)
result_label.pack(pady=30)

root.mainloop()
