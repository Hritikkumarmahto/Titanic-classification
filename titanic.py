import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
#from sklearn.ensemble import RandomForestClassifier
#Load the training dataset
train_data = pd.read_csv("/content/Titanic-Dataset.csv")

# Preprocess the data
train_data = pd.get_dummies(train_data, columns=['Sex', 'Embarked'], drop_first=True)


# Explore the data with some plots
sns.countplot(x='Survived', data=train_data)
plt.title('Survival Count')
plt.show()

plt.figure(figsize=(10, 6))
sns.heatmap(train_data.corr(), annot=True, cmap='coolwarm')
plt.title('Correlation Heatmap')
plt.show()
# Split the data
X = train_data.drop("Survived", axis=1)
y = train_data["Survived"]
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
# Create and train the model
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)  # Train the model

# Make predictions on the validation set
y_pred = model.predict(X_val)

# Calculate accuracy
accuracy = accuracy_score(y_val, y_pred)
print(f"Accuracy: {accuracy:.2f}")

# Generate a classification report
report = classification_report(y_val, y_pred)
print("Classification Report:")
print(report)

# Generate and plot a confusion matrix
conf_matrix = confusion_matrix(y_val, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()
