import io
import tkinter as tk
from PIL import Image, ImageOps
import numpy as np

class DigitDrawApp:
    def __init__(self, master, model):
        self.master = master
        self.canvas = tk.Canvas(master, width=300, height=300, bg='white')
        self.canvas.pack()
        self.model = model
        
        # Bind mouse events
        self.canvas.bind("<B1-Motion>", self.draw)
        
        self.predict_button = tk.Button(master, text="Predict", command=self.make_prediction)
        self.predict_button.pack()
        
        self.clear_button = tk.Button(master, text="Clear", command=self.clear_canvas)
        self.clear_button.pack()
        
        self.result_label = tk.Label(master, text="Draw a digit and click Predict")
        self.result_label.pack()

    def draw(self, event):
        # Draw a small circle to simulate pen stroke
        r = 10
        self.canvas.create_oval(event.x-r, event.y-r, event.x+r, event.y+r, fill='black')
    
    def clear_canvas(self):
        self.canvas.delete("all")

    def make_prediction(self):
        
        # Generate a PostScript string of the canvas content
        ps_data = self.canvas.postscript(colormode='color')

        # Convert the PostScript data into a PIL Image (in-memory)
        img = Image.open(io.BytesIO(ps_data.encode('utf-8')))

        # Convert to grayscale and resize to match model input 8x8
        img = ImageOps.grayscale(img)
        img = ImageOps.invert(img)
        img = img.resize((8, 8))
        # img.save("output.jpg")

        # Convert to NumPy and flatten
        arr = np.array(img).reshape(1, -1)

        # Predict using your MLP
        prediction = self.model.predict(arr)

        # Display the result
        self.result_label.config(text=f"Predicted digit: {prediction[0]}")