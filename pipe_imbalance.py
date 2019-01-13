import pandas as pd
import numpy as np
from imblearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.combine import SMOTEENN
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV,StratifiedKFold
from sklearn.linear_model import LogisticRegression



def model_classification(verbose):
    # simple classification
    from sklearn.linear_model import LogisticRegression
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.naive_bayes import BernoulliNB
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.neural_network import MLPClassifier
    
    # hard classification
    from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
    
    if verbose:
        print("Set Simple Model")
    logreg = LogisticRegression(random_state = 123)
    knn = KNeighborsClassifier()
    nb = BernoulliNB()
    dt = DecisionTreeClassifier(random_state = 123)
    
    if verbose:print("Set Hard Model\n")
    rf = RandomForestClassifier(random_state = 123)
    gbc = GradientBoostingClassifier(random_state = 123)
    mlp = MLPClassifier(random_state = 123)
    
    model = [logreg,knn,nb,dt,rf,gbc,mlp]
    method_name = ['Logistic Regression',
                  'K-Nearest Neighbor',
                  'Naive Bayes',
                  'Decision Tree',
                  'Random Forest',
                  'Gradient Boosting Classifier',
                  'Multilayer Perceptron']
    return [model,method_name]
    
def pipe_imbalance(X,y,imb = RandomUnderSampler(),verbose = False):
    
    df_eval = pd.DataFrame(columns = ['Model',
                                    'Accuracy',
                                    'Precision',
                                    'Recall',
                                    'AUC',
                                    'F1_score',
                                    'Log_loss',
                                    'Time'])

    if verbose: print("Split Training and Testing\n")
    X_train, X_test,y_train,y_test = train_test_split(X,
                                                        y,
                                                        test_size = 0.3,
                                                        stratify = y,
                                                        random_state = 123)

    if verbose: print('Import Classification Method\n')
    
    list_model = model_classification(True)
    df_eval['Model']=list_model[1]

    if verbose: print('Building Pipeline\n')
    pipe = Pipeline([('imb',imb),('classifier',LogisticRegression())])

    if verbose: print('Defining Params and Scoring\n')
    params = {'classifier': list_model[0]}
    scorers = ['accuracy','precision','recall','roc_auc','f1','neg_log_loss']
    
    skf = StratifiedKFold(n_splits=10,random_state = 123)
    
    grid = GridSearchCV(estimator = pipe,
                    param_grid = params,
                    scoring = scorers,
                    refit = 'accuracy',
                    cv = skf)
    grid.fit(X_train,y_train)
    
    df_eval['Accuracy']  = grid.cv_results_['mean_test_accuracy']
    df_eval['Precision'] = grid.cv_results_['mean_test_precision']
    df_eval['Recall'] = grid.cv_results_['mean_test_recall']
    df_eval['AUC'] = grid.cv_results_['mean_test_roc_auc']
    df_eval['F1_score'] = grid.cv_results_['mean_test_f1']
    df_eval['Log_loss'] = grid.cv_results_['mean_test_neg_log_loss']
    df_eval['Time'] = grid.cv_results_['mean_fit_time']
    
    return [grid,df_eval,X_test,y_test]
