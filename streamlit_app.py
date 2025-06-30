import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
from PIL import Image
# Define constants
IMG_SIZE = (224, 224)  
CLASS_MAP = {"with_mask": 0, "without_mask": 1, "mask_weared_incorrect": 2}
NUM_CLASSES = len(CLASS_MAP)

# Define loss functions for model loading
def box_loss(y_true, y_pred):
    # y_true and y_pred are tensors for boxes_output_reshape: (batch_size, 10, 4)
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)
    
    # Debugging: Print shapes and dtypes
    tf.print("box_true shape:", tf.shape(y_true), "dtype:", y_true.dtype)
    tf.print("box_pred shape:", tf.shape(y_pred), "dtype:", y_pred.dtype)
    
    # MSE loss
    loss = tf.reduce_mean(tf.square(y_true - y_pred), axis=[1, 2])
    return loss

def class_loss(y_true, y_pred):
    # y_true: (batch_size, 10), y_pred: (batch_size, 10, 3)
    y_true = tf.cast(y_true, tf.int32)
    y_pred = tf.cast(y_pred, tf.float32)
    
    # Debugging: Print shapes and dtypes
    tf.print("class_true shape:", tf.shape(y_true), "dtype:", y_true.dtype)
    tf.print("class_pred shape:", tf.shape(y_pred), "dtype:", y_pred.dtype)
    
    # Convert to one-hot and apply mask for padded objects
    valid_mask = tf.cast(y_true != 0, tf.float32)
    y_true_one_hot = tf.one_hot(y_true, depth=NUM_CLASSES)
    loss = tf.keras.losses.categorical_crossentropy(y_true_one_hot, y_pred)
    loss = tf.reduce_mean(loss * valid_mask, axis=1)
    return loss
    
def streamlit_app():
    st.write("Debug: Streamlit app started")
    st.title("Face Mask Detection")
    uploaded_file = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg"], key="image_uploader")
    
    if uploaded_file is not None:
        st.write("Debug: Image uploaded successfully")
        try:
            model = tf.keras.models.load_model(
                "face_mask_model.keras",
                custom_objects={"box_loss": box_loss, "class_loss": class_loss}
            )
            st.write("Debug: Model loaded successfully")
        except Exception as e:
            st.error(f"Error loading model: {str(e)}")
            return
        
        try:
            image = Image.open(uploaded_file)
            # Convert RGBA to RGB if necessary
            if image.mode == "RGBA":
                image = image.convert("RGB")
            image_resized = image.resize(IMG_SIZE)
            img_array = tf.keras.preprocessing.image.img_to_array(image_resized) / 255.0
            st.write(f"Debug: img_array shape: {img_array.shape}, dtype: {img_array.dtype}")
            img_array = np.expand_dims(img_array, axis=0)
            st.write(f"Debug: Input shape to model: {img_array.shape}")
            
            # Predict
            pred_boxes, pred_labels = model.predict(img_array)
            st.write(f"pred_boxes shape: {pred_boxes.shape}, pred_labels shape: {pred_labels.shape}")
            st.write(f"pred_labels max confidence: {np.max(pred_labels[0], axis=-1)}")
            
            # Convert image to BGR for OpenCV processing
            img = np.array(image_resized)
            img = img.astype(np.uint8)  # Ensure uint8 dtype
            
            # Draw predictions
            boxes_drawn = 0
            for i in range(len(pred_boxes[0])):
                if np.max(pred_labels[0][i]) > 0.5:
                    box = pred_boxes[0][i] * np.array([IMG_SIZE[1], IMG_SIZE[0], IMG_SIZE[1], IMG_SIZE[0]])
                    img = cv2.rectangle(img, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (255, 0, 0), 2)
                    label = np.argmax(pred_labels[0][i])
                    label_text = list(CLASS_MAP.keys())[list(CLASS_MAP.values()).index(label)]
                    img = cv2.putText(img, label_text, (int(box[0]), int(box[1])-10), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
                    boxes_drawn += 1
            st.write(f"Number of boxes drawn: {boxes_drawn}")
            
            # Convert BGR to RGB for Streamlit display
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            # Display image
            st.image(img, caption="Predicted Image", use_column_width=True)
        except Exception as e:
            st.error(f"Error processing image: {str(e)}")
    else:
        st.write("Debug: No image uploaded yet")
if __name__ == "__main__":
    import sys
    if "streamlit" in sys.modules and sys.argv[0].endswith("streamlit_app.py"):
        print("Starting Streamlit app...")
        streamlit_app()
    else:
        print("To run the Streamlit app, use: `streamlit run streamlit_app.py`")
