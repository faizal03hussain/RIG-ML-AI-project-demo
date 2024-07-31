
# Job Skill Predictor

## Overview

The Job Skill Predictor is a Streamlit application designed to predict the skills required for a given job description. Users can upload job descriptions in text, image, or PDF formats, and the application will extract the text and predict the necessary skills using a pre-trained machine learning model.

## Features

- **Text Input**: Allows users to paste job description text directly into the app.
- **Image Input**: Extracts text from uploaded images using Tesseract OCR.
- **PDF Input**: Extracts text from uploaded PDF files using `pdfplumber`.
- **Skill Prediction**: Uses a trained logistic regression model to predict skills based on the extracted text.

## Getting Started

### Prerequisites

- Python 3.6+
- Required Python libraries:
  - `streamlit`
  - `pandas`
  - `numpy`
  - `pytesseract`
  - `pdfplumber`
  - `Pillow`
  - `joblib`
  - `scikit-learn`

### Installation

1. **Clone the Repository**

   ```bash
   git clone https://github.com/faizal03hussain/RIG-ML-AI-project-demo.git
   cd job-skill-predictor
   ```

2. **Install Dependencies**

   It is recommended to use a virtual environment. You can create one using `venv` or `virtualenv`, and then install the required packages.

   ```bash
   python -m venv env
   source env/bin/activate  # On Windows use `env\Scripts\activate`
   pip install -r requirements.txt
   ```

   Create a `requirements.txt` file with the following contents:

   ```
   streamlit
   pandas
   numpy
   pytesseract
   pdfplumber
   Pillow
   joblib
   scikit-learn
   ```

3. **Train and Save the Model**

   Run the training script to generate and save the model and vectorizer.

   ```bash
   python project.ipynb
   ```

### Usage

1. **Run the Streamlit App**

   Start the Streamlit app with the following command:

   ```bash
   streamlit run app.py
   ```

2. **Interact with the App**

   Open your web browser and navigate to `http://localhost:8501` to use the app. You will see options to upload job descriptions in text, image, or PDF formats.

   - **Text**: Enter the job description directly into the provided text area and click "Predict Skills".
   - **Image**: Upload an image containing job description text. The app will extract the text and predict the skills.
   - **PDF**: Upload a PDF file containing job description text. The app will extract the text and predict the skills.
