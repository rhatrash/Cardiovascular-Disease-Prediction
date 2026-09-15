from pathlib import Path
import numpy as np,pandas as pd,joblib
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score,roc_auc_score,confusion_matrix
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
BASE=Path('/home/ubuntu/cardio_project'); OUT=BASE/'outputs'; FIG=BASE/'figures'
df=pd.read_csv(BASE/'data/cardio_train.csv',sep=';'); df['age_years']=df.age/365.25; df=df.drop_duplicates(); df=df[df.height.between(120,220)&df.weight.between(30,250)&df.ap_hi.between(60,250)&df.ap_lo.between(40,200)&(df.ap_hi>=df.ap_lo)].copy(); df['bmi']=df.weight/(df.height/100)**2
features=['age_years','gender','height','weight','ap_hi','ap_lo','cholesterol','gluc','smoke','alco','active','bmi']; target='cardio'; sample=df.groupby(target,group_keys=False).sample(n=6000,random_state=42); X=sample[features]; y=sample[target]; Xtr,Xte,ytr,yte=train_test_split(X,y,test_size=.2,random_state=42,stratify=y)
models={'SVM':Pipeline([('imp',SimpleImputer(strategy='median')),('sc',StandardScaler()),('m',SVC(C=1,probability=True,random_state=42))]),'KNN':Pipeline([('imp',SimpleImputer(strategy='median')),('sc',StandardScaler()),('m',KNeighborsClassifier(n_neighbors=15))]),'Decision Tree':Pipeline([('imp',SimpleImputer(strategy='median')),('m',DecisionTreeClassifier(max_depth=8,min_samples_leaf=10,class_weight='balanced',random_state=42))]),'Logistic Regression':Pipeline([('imp',SimpleImputer(strategy='median')),('sc',StandardScaler()),('m',LogisticRegression(max_iter=500,class_weight='balanced',random_state=42))]),'Random Forest':Pipeline([('imp',SimpleImputer(strategy='median')),('m',RandomForestClassifier(n_estimators=150,max_depth=12,min_samples_leaf=4,class_weight='balanced',n_jobs=-1,random_state=42))])}
rows=[]; fitted={}
for name,m in models.items():
 m.fit(Xtr,ytr); fitted[name]=m; p=m.predict(Xte); pr=m.predict_proba(Xte)[:,1]; rows.append({'Model':name,'Test accuracy':accuracy_score(yte,p),'Precision':precision_score(yte,p),'Recall':recall_score(yte,p),'F1':f1_score(yte,p),'ROC-AUC':roc_auc_score(yte,pr)}); pd.DataFrame(confusion_matrix(yte,p),index=['Actual 0','Actual 1'],columns=['Predicted 0','Predicted 1']).to_csv(OUT/f'confusion_matrix_{name.lower().replace(" ","_")}.csv')
res=pd.DataFrame(rows).sort_values('Test accuracy',ascending=False); res.to_csv(OUT/'model_comparison.csv',index=False); best=res.iloc[0].Model; joblib.dump({'model':fitted[best],'features':features,'best_model':best},OUT/'cardiovascular_disease_detector.joblib'); pd.Series({'modeling_sample_rows':len(sample),'best_model':best,'best_test_accuracy':float(res.iloc[0]['Test accuracy']),'full_clean_rows':len(df)}).to_json(OUT/'run_summary.json',indent=2); print(res.to_string(index=False))
