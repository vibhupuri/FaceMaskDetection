import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
from PIL import Image

def streamlit_app():
    st.title("Face Mask Detection")
    uploaded_file = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg"])
    
    if uploaded_file is not None:
        model = tf.keras.models.load_model(
            "face_mask_model.keras",
            custom_objects={"box_loss": box_loss, "class_loss": class_loss}
        )
        image = Image.open(uploaded_file)
        image_resized = image.resize(IMG_SIZE)
        img_array = tf.keras.preprocessing.image.img_to_array(image_resized) / 255.0
        img_array = np.expand_dims(img_array, axis=0)
        
        pred_boxes, pred_labels = model.predict(img_array)
        img = np.array(image_resized)
        
        for i in range(len(pred_boxes[0])):
            if np.max(pred_labels[0][i]) > 0.5:
                box = pred_boxes[0][i] * np.array([IMG_SIZE[1], IMG_SIZE[0], IMG_SIZE[1], IMG_SIZE[0]])
                img = cv2.rectangle(img, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (255, 0, 0), 2)
                label = np.argmax(pred_labels[0][i])
                label_text = list(CLASS_MAP.keys())[list(CLASS_MAP.values()).index(label)]
                img = cv2.putText(img, label_text, (int(box[0]), int(box[1])-10), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
        
        st.image(img, caption="Predicted Image", use_column_width=True)
if __name__ == "__main__":
    print("Starting Streamlit app...")
    streamlit_app()
