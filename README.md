# ML-Assignment-4---Classification-Problem
This project explains about the Classification machine learning models
# Key components to be fulfilled:

# 1.Loading and Preprocessing:
Load the breast cancer dataset from sklearn.
Preprocess the data to handle any missing values and perform necessary feature scaling.
Explain the preprocessing steps you performed and justify why they are necessary for this dataset.

import warnings
warnings.filterwarnings("ignore")

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report 


