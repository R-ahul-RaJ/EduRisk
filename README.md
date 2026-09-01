# EduRisk – Student Performance Risk Prediction using ML & XAI

EduRisk is a machine learning-based student performance risk prediction system developed to identify students who may be academically at risk based on various student-related factors.

The project combines Machine Learning with Explainable AI (XAI) to provide predictions while making the model's results easier to understand and interpret.

## Project Overview

The system uses the **Student Performance Factors** dataset to analyze different factors that can influence student performance and classify students into two categories:

- **At Risk**
- **Not At Risk**

A performance score threshold of **70** is used for risk classification.

The trained machine learning model is integrated into a **Flask web application**, allowing users to enter student information and receive a predicted risk category through an interactive web interface.

## Key Features

- Exploratory Data Analysis (EDA)
- Data cleaning and preprocessing
- Feature engineering
- Machine learning model training and evaluation
- Student performance risk prediction
- Explainable AI (XAI)
- Interactive Flask web application
- User-based prediction interface
- Compare Prediction Records
- PDF Report Generation

## Technologies Used

- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- Explainable AI (XAI)
- Flask
- HTML
- CSS
- Jupyter Notebook

## Dataset

The project uses the **Student Performance Factors** dataset.

The dataset contains information related to various factors affecting student performance, including:

- Hours studied
- Attendance
- Previous scores
- Sleep
- Motivation
- Access to resources
- Family involvement
- Parental involvement
- Peer influence
- Internet access
- Physical activity
- Learning disabilities
- And other student-related factors

## Project Structure

```text
EduRisk/
│
├── Models/                         # Trained machine learning models
├── static/                         # Static files for the web application
├── templates/                      # HTML templates for the Flask application
├── app.py                          # Flask application
├── StudentPerformanceFactors.csv   # Dataset
├── StudentRisk.ipynb               # Model development and analysis
├── requirements.txt                # Python dependencies
├── .gitignore
└── README.md

## Running the Project Locally

### 1. Clone the repository

```bash
git clone https://github.com/YOUR_USERNAME/EduRisk.git
cd EduRisk

### 2. Install the required dependencies

```bash
pip install -r requirements.txt

### 3. Run the Flask application

```bash
python app.py

Open the local URL displayed in the terminal in your web browser.