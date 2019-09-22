import numpy as np
import pandas as pd
import os
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier 
from sklearn.externals import joblib
fileName='finalizedModel.model'

def MatchModel(model,data,features,result,test):
    
    model.fit(data[features],data[result])
    predictions=model.predict(data[features])
    accuracy=metrics.accuracy_score(predictions, data[result])
    print('Train Data Accuracy : %s' % '{0:.3%}'.format(accuracy))
    predict_test=model.predict(test[features])
    accuracy_test=metrics.accuracy_score(predict_test,test[result])
    accuracy_load_model=0.0
    if os.path.exists(fileName):
        loaded_model=joblib.load(fileName)
        load_model_predict=loaded_model.predict(test[features])
        accuracy_load_model=metrics.accuracy_score(load_model_predict, test[result])
        if accuracy_test>accuracy_load_model:
            joblib.dump(model, fileName)
            print('Test Data Accuracy : %s' % '{0:.3%}'.format(accuracy_test))
        else:
            print('Test Data Accuracy : %s' % '{0:.3%}'.format(accuracy_load_model))
    else:
        joblib.dump(model, fileName)
        print('Test Data Accuracy : %s' % '{0:.3%}'.format(accuracy_test))

def main():
    df=pd.read_csv("train.csv")
    df_test=pd.read_csv("test.csv")
    df_test.columns=df.columns[0:].tolist()
    teams=list(set(df['Team 1'].tolist()+ df['Team 2'].tolist()))
    t_teams=list(set(df_test['Team 1'].tolist()+ df_test['Team 2'].tolist()))
    
    set1=list(set(teams+t_teams))
    teams=np.sort(set1)
    cities=np.sort(list(set(list(df['City'])+list(df_test['City']))))
    cityDict={k:v+1 for v,k in enumerate(cities)}
    dict={k:v+1 for v,k in enumerate(teams)}
    encode={'Team 1':dict,'Team 2':dict,'City':cityDict}
    df.replace(encode,inplace=True)
    df_test.replace(encode,inplace=True)
    features=df.columns[1:-1].tolist()
    features.remove('DateOfGame')
    features.remove('TimeOfGame')
    model=RandomForestClassifier(90)
    result_var=['Winner (team 1=1, team 2=0)']
    MatchModel(model,df,features,result_var,df_test)
    

    
main()

