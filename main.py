import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the Titanic dataset
titanic_data = pd.read_csv('titanic.csv')

# Data Cleaning

# Check for missing values
missing_values = titanic_data.isnull().sum()
print("Missing values before cleaning:")
print(missing_values)

# Handle missing values
titanic_data['Age'].fillna(titanic_data['Age'].median(), inplace=True)
titanic_data['Embarked'].fillna(titanic_data['Embarked'].mode()[0], inplace=True)
titanic_data.drop(columns=['Cabin'], inplace=True)

# Verify that there are no more missing values
missing_values_after_cleaning = titanic_data.isnull().sum()
print("\nMissing values after cleaning:")
print(missing_values_after_cleaning)

# Exploratory Data Analysis (EDA)

# Summary Statistics
summary_stats = titanic_data.describe()
print("\nSummary Statistics:")
print(summary_stats)

# Set the style for the plots
sns.set(style="whitegrid")

# Survival rate by class
plt.figure(figsize=(10, 6))
sns.barplot(x='Pclass', y='Survived', data=titanic_data)
plt.title('Survival Rate by Class')
plt.ylabel('Survival Rate')
plt.xlabel('Class')
plt.show()

# Survival rate by sex
plt.figure(figsize=(10, 6))
sns.barplot(x='Sex', y='Survived', data=titanic_data)
plt.title('Survival Rate by Sex')
plt.ylabel('Survival Rate')
plt.xlabel('Sex')
plt.show()

# Survival rate by age
plt.figure(figsize=(10, 6))
sns.histplot(data=titanic_data, x='Age', hue='Survived', multiple='stack')
plt.title('Survival Rate by Age')
plt.ylabel('Count')
plt.xlabel('Age')
plt.show()

# Survival rate by fare
plt.figure(figsize=(10, 6))
sns.histplot(data=titanic_data, x='Fare', hue='Survived', multiple='stack', bins=30)
plt.title('Survival Rate by Fare')
plt.ylabel('Count')
plt.xlabel('Fare')
plt.show()
