#!/usr/bin/env python
# coding: utf-8

# # Data Preprocessing

# In[1]:


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# Load the data
data = pd.read_csv('C:/Siddhant/CodeAlpha ML 3 Months/Task 5/data.csv')

# Drop rows with any missing values
data.dropna(inplace=True)

# Encode categorical variables
label_encoders = {}
for col in ["Age","Sex","Job", "Housing", "Saving accounts", "Checking account", "Credit amount", "Duration", "Purpose", "Risk Level"]:
    le = LabelEncoder()
    data[col] = le.fit_transform(data[col])
    label_encoders[col] = le

# Split data into features and target
X = data.drop(columns=["Risk Level"])
y = data["Risk Level"]

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# # 1 Logistic Regression

# In[2]:


from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Initialize the model
logreg_model = LogisticRegression(random_state=42)

# Train the model
logreg_model.fit(X_train, y_train)

# Predict on the test data
y_pred_logreg = logreg_model.predict(X_test)

# Evaluate the model
accuracy_logreg = accuracy_score(y_test, y_pred_logreg)
print("Logistic Regression Accuracy:", accuracy_logreg)


# # Decision Tree Classifier

# In[3]:


from sklearn.tree import DecisionTreeClassifier

# Initialize the model
dt_model = DecisionTreeClassifier(random_state=42)

# Train the model
dt_model.fit(X_train, y_train)

# Predict on the test data
y_pred_dt = dt_model.predict(X_test)

# Evaluate the model
accuracy_dt = accuracy_score(y_test, y_pred_dt)
print("Decision Tree Accuracy:", accuracy_dt)


# # Random Forest Classifier

# In[4]:


from sklearn.ensemble import RandomForestClassifier

# Initialize the model
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)

# Train the model
rf_model.fit(X_train, y_train)

# Predict on the test data
y_pred_rf = rf_model.predict(X_test)

# Evaluate the model
accuracy_rf = accuracy_score(y_test, y_pred_rf)
print("Random Forest Accuracy:", accuracy_rf)


# # Gradient Boost

# In[5]:


from sklearn.ensemble import GradientBoostingClassifier

# Initialize the model
gb_model = GradientBoostingClassifier(random_state=42)

# Train the model
gb_model.fit(X_train, y_train)

# Predict on the test data
y_pred_gb = gb_model.predict(X_test)

# Evaluate the model
accuracy_gb = accuracy_score(y_test, y_pred_gb)
print("Gradient Boosting Accuracy:", accuracy_gb)


# # Support Vector Macine (SVM) 

# In[6]:


from sklearn.svm import SVC

# Initialize the model
svm_model = SVC(random_state=42)

# Train the model
svm_model.fit(X_train, y_train)

# Predict on the test data
y_pred_svm = svm_model.predict(X_test)

# Evaluate the model
accuracy_svm = accuracy_score(y_test, y_pred_svm)
print("Support Vector Machine Accuracy:", accuracy_svm)


# # Model Selection

# In[7]:


# Create a dictionary to store model accuracies
model_accuracies = {
    "Logistic Regression": accuracy_logreg,
    "Decision Tree": accuracy_dt,
    "Random Forest": accuracy_rf,
    "Gradient Boosting": accuracy_gb,
    "Support Vector Machine": accuracy_svm
}

# Find the best model and its accuracy
best_model = max(model_accuracies, key=model_accuracies.get)
best_accuracy = model_accuracies[best_model]

# Print the best model and its accuracy
print("Best Model:", best_model)
print("Accuracy:", best_accuracy)


# In[ ]:




