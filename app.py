import matplotlib.pyplot as plt
import numpy as np
import streamlit as st
import tensorflow as tf

st.title("Potato Disease classification")

col1, col2 = st.columns(2, gap="large")


def predict(image):
    class_name = [
        "Potato Early Blight",
        "Potato Late Blight",
        "Potato Healthy",
    ]
    model = tf.keras.models.load_model("data/model/trained_model.h5")
    img_array = tf.keras.preprocessing.image.img_to_array(image)
    img_array = tf.expand_dims(img_array, 0)

    prediction = model.predict(img_array)
    print(f"prediction: {prediction}")

    predicted_class = class_name[np.argmax(prediction[0])]
    confidence = round(100 * (np.max(prediction[0])), 2)

    return predicted_class, confidence


with col1:
    st.write("Upload image to classify...")
    uploaded_file = st.file_uploader(label="Upload Image", type=["jpeg", "jpg", "png"])
    if uploaded_file is not None:
        # Convert the file to an opencv image.
        image = plt.imread(uploaded_file)

        # Now do something with the image! For example, let's display it:
        st.image(image, channels="RGB")


with col2:
    if uploaded_file is not None:
        if st.button(label="Classify Image", type="primary"):
            cls, score = predict(image=image)
            st.write(
                f"Resulted class is: **{cls}** with the confidence scores of **{str(score)}** %"
            )
    else:
        st.subheader("Please upload image first :pensive:")