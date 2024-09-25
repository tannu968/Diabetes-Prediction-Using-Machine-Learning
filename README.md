# Diabetes-Prediction-Using-Machine-Learning

This project aims to predict diabetes outcomes using various health metrics. The dataset contains information about different health parameters, and the goal is to build a predictive model to estimate the likelihood of diabetes.

## Table of Contents
- [Installation](#installation)
- [Usage](#usage)
- [Data Description](#data-description)
- [Model Explanation](#model-explanation)
- [Results](#results)
- [Contributing](#contributing)
- [License](#license)

## Installation

To run this project, you need to have Python installed along with the necessary libraries. You can install the required packages using pip:

```bash
pip install pandas matplotlib scikit-learn

Usage
Clone the repository or download the project files.
Make sure you have the dataset diabetes.csv located in the specified path (E:/diabetes.csv).

Run the script:
python main.py

Data Description
The dataset diabetes.csv contains the following columns:

##Pregnancies:
 Number of pregnancies
Glucose: Glucose level

##BloodPressure:
 Blood pressure value

##SkinThickness:
Thickness of skin

##Insulin:
Insulin level

##BMI:
Body Mass Index

##DiabetesPedigreeFunction:
 Diabetes pedigree function

Age: Age of the individual
Outcome: Diabetes outcome (0 = No, 1 = Yes)
Model Explanation
The models used in this project include:

##Decision Tree Classifier:
Used to classify the diabetes outcome based on the input features.
Bagging Classifier: An ensemble method that combines multiple decision trees for better performance.
Random Forest Classifier: Another ensemble method that uses a collection of decision trees to improve classification accuracy.
Results
The script prints the accuracy scores for the Decision Tree, Bagging, and Random Forest models based on cross-validation and test sets.



