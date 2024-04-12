# Using Python Interpreter 3.9.6
from sklearn.datasets import load_iris
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler

class IrisClassifier():
    def __init__(self):
        self.__nbClassifier = GaussianNB()
        self.__svmClassifier = SVC(probability=True)
        self.__rfClassifier = RandomForestClassifier()
        self.__xgbClassifier = XGBClassifier()
        self.__knnClassifier = KNeighborsClassifier()

    # Print the performance metrics
    def printPerformanceMetrics(self, metrics):
        xgb_flag = False

        # Calculate each model's results and print the metrics
        # If no metrics are printed, default values were used
        for k, v in metrics.items():
            if 'accuracy' in k.split()[-1]:
                print('\t\t', ' '.join(k.split()[:-1]), '\n')

            # Print the score and best hyperparameters for each metric if it is not XGBoost
            if not 'XGBoost' in k[:-1] or xgb_flag:
                if 'macro' in k.split()[-1]:
                    print(f'F1-Score: {v.best_score_:0.04f}'.ljust(20) + f'Best Hyperparameters: {v.best_estimator_}\n')
                elif 'roc' in k.split()[-1]:
                    print(f'ROC AUC: {v.best_score_:0.04f}'.ljust(20) + f'Best Hyperparameters: {v.best_estimator_}\n\n')
                else:
                    print(f'Accuracy: {v.best_score_:0.04f}'.ljust(20) + f'Best Hyperparameters: {v.best_estimator_}\n')

            # Print each metric for XGBoost
            # Hyperparameters not printed for XGBoost because best_estimator_ displays 20+ parameters (most default)
            # To see them, change the xgb_flag to True
            else:
                if 'macro' in k.split()[-1]:
                    print(f'F1-Score: {v.best_score_:0.04f}\n')
                elif 'roc' in k.split()[-1]:
                    print(f'ROC AUC: {v.best_score_:0.04f}\n\n')
                else:
                    print(f'Accuracy: {v.best_score_:0.04f}\n')
        
    # Train the model using GridSearchCV to identify the best hyperparameters for each metric for each model
    # and fit each model on the training data
    def train(self, X_train, y_train):
        metrics = {}
        scores = ['accuracy', 'f1_macro', 'roc_auc_ovo']

        # Parameter grids for each model
        # NB parameter grid
        naive_bayes = {
            'var_smoothing': [1e-9, 1e-7, 1e-8, 1e-10, 1e-11]
        }
        
        # SVM parameter grid
        support_vector_machine = {
            'C': [1, 6, 10, 14],
            'gamma':  [0.01, 0.1, 1],
            'kernel': ['rbf', 'linear', 'poly', 'sigmoid']
        }

        # RF parameter grid
        random_forest = {
            'n_estimators': [60, 80, 100],
            'criterion': ['gini', 'entropy'],
            'min_samples_split': [2, 3, 4],
            'max_features': ['sqrt', None]
        }

        # XBG parameter grid
        xgb_param_grid = {
            'max_depth': [1, 2, 3, 4],
            'n_estimators': [20, 40, 60, 80],
            'learning_rate': [0.005, 0.01, 0.05]
        }

        # KNN parameter grid
        knn_param_grid = {
            'n_neighbors': [1, 3, 5, 7, 9],
            'weights': ['uniform', 'distance'],
            'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute']
        }
        
        # Initialize the grid list to be passed to GridSearchCV
        grids = [
            [self.__nbClassifier, naive_bayes, 'Naive Bayes '],
            [self.__svmClassifier, support_vector_machine, 'Support Vector Machine '],
            [self.__rfClassifier, random_forest, 'Random Forest '],
            [self.__xgbClassifier, xgb_param_grid, 'XGBoost '],
            [self.__knnClassifier, knn_param_grid, 'K-Nearest Neighbor ']
        ]

        # Call GridSearchCV for NB, SVM, RF and fit each metric and save it in the dictionary, metrics
        for grid in range(len(grids)):
            for score in range(len(scores)):
                metrics[str(grids[grid][2]) + scores[score]] = GridSearchCV(grids[grid][0], grids[grid][1], cv=5, scoring=scores[score], return_train_score=True).fit(X_train, y_train)

        # Print the results
        self.printPerformanceMetrics(metrics)


def main():
    iris = IrisClassifier()

    data = load_iris()
    labelData = data['data']
    classData = data['target']

    # Split the data into training and testing
    X_train, X_test, y_train, y_test = train_test_split(labelData, classData, test_size=0.2, random_state=5)

    # Preprocessing
    # Use the StandardScaler to transform the train and test data
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)

    # Pass the training data
    iris.train(X_train_scaled, y_train)

if __name__ == '__main__':
    main()
