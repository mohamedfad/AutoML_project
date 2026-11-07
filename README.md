# AutoML Project

## Overview

AutoML is a web-based application built with Python and Streamlit. It aims to simplify the Machine Learning workflow through an interactive and user-friendly interface.

The application allows users to upload their own datasets, preprocess the data, train Machine Learning models, and evaluate their performance without manually implementing every step.

## Demo Video

Watch the project demo:

[▶ Watch AutoML Project Demo](https://github.com/user-attachments/assets/9a3f8ab7-500e-452d-8d6d-318fde3bd055)

## Key Features

- Upload custom datasets.
- Select the target column.
- Preprocess and prepare the data.
- Support for Regression and Classification tasks.
- Train and evaluate Machine Learning models.
- Compare model performance.
- Automatically select the best-performing model.
- Hyperparameter Tuning to optimize model performance.

## Technologies Used

- Python
- Streamlit
- Pandas
- NumPy
- Scikit-learn

## Project Structure

```text
AutoML_project/
│
├── app.py
├── eda
├── model
├── preprocessing.py
├── tunning.py
├── requirements.txt
├── .gitignore
└── README.md
```

## Installation and Setup

### 1. Clone the Repository

```bash
git clone https://github.com/mohamedfad/AutoML_project.git
```

### 2. Navigate to the Project Directory

```bash
cd AutoML_project
```

### 3. Create a Virtual Environment

```bash
python -m venv .venv
```

### 4. Activate the Virtual Environment

On Windows:

```bash
.venv\Scripts\activate
```

### 5. Install the Required Dependencies

```bash
pip install -r requirements.txt
```

### 6. Run the Application

```bash
streamlit run app.py
```



