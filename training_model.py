import pandas as pd
import joblib
import warnings
warnings.filterwarnings("ignore", message="The number of unique classes is greater than 50%")

from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
#from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import (
    RandomForestClassifier,
   # BaggingClassifier,
   # GradientBoostingClassifier,
   # AdaBoostClassifier,
   # ExtraTreesClassifier
)

#Dataset_Loading:
crop = pd.read_csv("Crop_recommendation.csv")

#Mapping_Crops_to_numbers
crop_dict={'rice': 1, 'maize': 2, 'jute': 3, 'cotton': 4, 'coconut': 5, 
           'papaya': 6, 'orange': 7, 'apple': 8, 'muskmelon': 9, 'watermelon': 10, 
           'grapes': 11, 'mango': 12, 'banana': 13, 'pomegranate': 14, 
           'lentil': 15, 'blackgram': 16, 'mungbean': 17, 'mothbeans': 18,
           'pigeonpeas': 19, 'kidneybeans': 20, 'chickpea': 21, 'coffee': 22
           }
crop["crop_num"] = crop["label"].map(crop_dict)

x = crop.drop(["crop_num", "label"], axis=1)
y = crop["crop_num"]
#Training_the_dataset 
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=74)

#Scaling_Features
scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

#Model_Evaluation
models={'Logistic Regression': LogisticRegression(max_iter=1000),
        'Naive Bayes': GaussianNB(),
        'Support Vector Machine': SVC(probability=True),
        'K-Nearest Neighbors': KNeighborsClassifier(),
      # 'Decision Tree': DecisionTreeClassifier(),
        'Random Forest': RandomForestClassifier(),
      # 'Bagging': BaggingClassifier(),
      # 'AdaBoost': AdaBoostClassifier(),
      # 'Gradient Boosting': GradientBoostingClassifier(),
      # 'Extra Trees': ExtraTreesClassifier()
    }
best_model = None
best_acc = 0
for name, model in models.items():
    model.fit(x_train, y_train)
    y_pred=model.predict(x_test)
    acc=accuracy_score(y_test, y_pred)
    print(f"{name}: {acc:.4f}")
    if(acc>best_acc):
        best_acc = acc
        best_model = model
print(f"\n 🎯 Best Model : {best_model.__class__.__name__} with accuracy: {best_acc:.4f} 🎯")

#Predictions_with_best_model
y_pred = best_model.predict(x_test)
print("\nConfusion Matrix of our Best Model:")
print(confusion_matrix(y_test, y_pred))
print("\nClassification Report of our Best Model:")
print(classification_report(y_test, y_pred))

#Saving_Best_Model_and_Scaler
joblib.dump(best_model, "crop_model.pkl")
joblib.dump(scaler, "scaler.pkl")
print(" Model and scaler saved.✅")
