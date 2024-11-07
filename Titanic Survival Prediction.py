import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score

# Load the dataset
titanic_data = pd.read_csv(r"C:\Users\Snehal\Downloads\titanic.csv")

# Display the first few rows
print(titanic_data.head())

# Data Cleaning
# Check for Missing Values
print(titanic_data.isnull().sum())

# Fill Missing Values
titanic_data = titanic_data.assign(
    Age=titanic_data['Age'].fillna(titanic_data['Age'].median()),
    Embarked=titanic_data['Embarked'].fillna(titanic_data['Embarked'].mode()[0])
)

# Feature Engineering
# Convert Categorical Variables
titanic_data['Sex'] = titanic_data['Sex'].map({'male': 0, 'female': 1})
titanic_data = pd.get_dummies(titanic_data, columns=['Embarked'], drop_first=True)

# Create New Features
titanic_data['FamilySize'] = titanic_data['SibSp'] + titanic_data['Parch']

# Exploratory Data Analysis (EDA)
# Visualize Data
sns.countplot(x='Survived', hue='Sex', data=titanic_data)
plt.title('Survival Rate by Gender')
plt.show()

# Model Building
# Select Features and Target
features = ['Pclass', 'Sex', 'Age', 'Fare', 'FamilySize', 'Embarked_Q', 'Embarked_S']
X = titanic_data[features]
y = titanic_data['Survived']

# Split the Data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the Model
model = LogisticRegression(max_iter=200)
model.fit(X_train, y_train)

# Model Evaluation
y_pred = model.predict(X_test)

# Evaluate the Model
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)

print(f'Accuracy: {accuracy:.2f}')
print(f'Precision: {precision:.2f}')
print(f'Recall: {recall:.2f}')
