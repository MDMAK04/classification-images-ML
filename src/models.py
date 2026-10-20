from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier  
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

def get_models(use_balanced=False):
    
    cw = "balanced" if use_balanced else None

    return {
        "SVM_RBF": Pipeline([
            ('scaler', StandardScaler()),
            ('model', SVC(kernel='rbf', class_weight=cw, random_state=42))
        ]),
        "SVM_Linear": Pipeline([
            ('scaler', StandardScaler()),
            ('model', SVC(kernel='linear', class_weight=cw, random_state=42))
        ]),
        "SVM_Sigmoid": Pipeline([
            ('scaler', StandardScaler()),
            ('model', SVC(kernel='sigmoid', class_weight=cw, random_state=42))
        ]),
        "KNN_k3": Pipeline([
            ('scaler', StandardScaler()),
            ('model', KNeighborsClassifier(n_neighbors=3))
        ]),
        "KNN_k5": Pipeline([
            ('scaler', StandardScaler()),
            ('model', KNeighborsClassifier(n_neighbors=5))
        ]),
        "Decision_Tree": Pipeline([
            ('scaler', StandardScaler()),
            ('model', DecisionTreeClassifier(class_weight=cw, random_state=42))
        ]),
        "Random_Forest": Pipeline([
            ('scaler', StandardScaler()),
            ('model', RandomForestClassifier(class_weight=cw, n_estimators=100, random_state=42))
        ])
    }