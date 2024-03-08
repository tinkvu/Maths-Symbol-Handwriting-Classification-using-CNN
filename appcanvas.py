import streamlit as st
from tensorflow import keras
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image
from streamlit_drawable_canvas import st_canvas

# Load the trained model
model = keras.models.load_model("model (1).h5")


# Get class names from the output layer
class_names = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'dot', 'minus', 'plus', 'slash', 'w', 'x', 'y', 'z']

def preprocess_image(img_array):
    # Ensure the image has 3 channels (RGB)
    img_array = img_array[:, :, :3]
    
    # Resize the image to target size
    img = Image.fromarray(img_array)
    img = img.resize((64, 64))
    img_array = np.array(img)
    img_array = img_array / 255.0  # Normalize the image
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

def predict(img_array):
    img_array = preprocess_image(img_array)
    prediction = model.predict(img_array)
    predicted_class = np.argmax(prediction)
    confidence = np.max(prediction) * 100
    return class_names[predicted_class], confidence

def main():
    st.title("Math Symbol Identification using CNN")
    st.write("The model is trained on 27,000 images of Math Symbols.")
    image_url = "symbols.gif"  # Replace with the URL of your image
    st.image(image_url,use_column_width=True)
    st.write("Try drawing any symbol on the canvas below:")
    
    
    
    # Create a drawing canvas
    canvas_result = st_canvas(
        fill_color="rgba(255, 165, 0, 0.3)",  # Initial drawing color
        stroke_width=5,
        stroke_color="rgb(0, 0, 0)",
        background_color="#fff",
        height=64,
        width=64,
        drawing_mode="freedraw",
        key="canvas",
    )

    if st.button("Predict"):
        if canvas_result.image_data is not None:
            # Make prediction
            class_name, confidence = predict(canvas_result.image_data)
            st.write(f"Prediction: {class_name}")
            st.write(f"Confidence: {confidence:.2f}%")
            

            
             # Add a button for reporting
            if st.button("Report Irrelevant Prediction"):
                st.write("Thank you for reporting! Our team will review the prediction.")

        else:
            st.warning("Please draw an image before predicting.")

    
# Run the Streamlit app
if __name__ == "__main__":
    main()
