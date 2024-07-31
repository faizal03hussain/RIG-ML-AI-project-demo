import streamlit as st
import pandas as pd
import numpy as np
import pytesseract
import pdfplumber
from PIL import Image
import joblib

# Load the trained model and vectorizer
model = joblib.load('skill_predictor_model.pkl')
vectorizer = joblib.load('vectorizer.pkl')

def extract_text_from_image(image):
    return pytesseract.image_to_string(image)

def extract_text_from_pdf(pdf_path):
    text = ""
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            text += page.extract_text()
    return text

def predict_skills(description):
    description_vectorized = vectorizer.transform([description])
    prediction = model.predict(description_vectorized)
    return prediction[0]

st.title('Job Skill Predictor')

# Input options
input_type = st.radio("Select input type", ["Text", "Image", "PDF"])

if input_type == "Text":
    job_description = st.text_area("Enter job description text")
    if st.button("Predict Skills"):
        if job_description:
            skills = predict_skills(job_description)
            st.write(f"Predicted Skills: {skills}")
        else:
            st.warning("Please enter a job description.")

elif input_type == "Image":
    uploaded_image = st.file_uploader("Upload an image file", type=["jpg", "jpeg", "png"])
    if uploaded_image:
        image = Image.open(uploaded_image)
        job_description = extract_text_from_image(image)
        st.text_area("Extracted Text", job_description, height=200)
        if st.button("Predict Skills"):
            if job_description:
                skills = predict_skills(job_description)
                st.write(f"Predicted Skills: {skills}")
            else:
                st.warning("No text found in the image.")

elif input_type == "PDF":
    uploaded_pdf = st.file_uploader("Upload a PDF file", type=["pdf"])
    if uploaded_pdf:
        job_description = extract_text_from_pdf(uploaded_pdf)
        st.text_area("Extracted Text", job_description, height=200)
        if st.button("Predict Skills"):
            if job_description:
                skills = predict_skills(job_description)
                st.write(f"Predicted Skills: {skills}")
            else:
                st.warning("No text found in the PDF.")
