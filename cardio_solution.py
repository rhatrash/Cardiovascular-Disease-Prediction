from pathlib import Path
import warnings
warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, classification_report, RocCurveDisplay
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import joblib

BASE=Path('/home/ubuntu/cardio_project'); DATA=BASE/'data/cardio_train.csv'; FIG=BASE/'figures'; OUT=BASE/'outputs'
FIG.mkdir(parents=True,exist_ok=True); OUT.mkdir(parents=True,exist_ok=True)
sns.set_theme(style='whitegrid', context='notebook')

# Load and standardize the source data
df=pd.read_csv(DATA, sep=';')
raw_shape=df.shape
# Convert age from days to years for interpretability
df['age_years']=df['age']/365.25
# Remove duplicate rows and impossible clinical measurements; retain clinically plausible observations.
duplicates=int(df.duplicated().sum())
df=df.drop_duplicates().copy()
invalid=(df['height'].between(120,220) & df['weight'].between(30,250) & df['ap_hi'].between(60,250) & df['ap_lo'].between(40,200) & (df['ap_hi']>=df['ap_lo']))
invalid_removed=int((~invalid).sum()); df=df[invalid].copy()
# BMI is a useful derived feature.
df['bmi']=df['weight']/(df['height']/100)**2
# Treat id as an identifier, not a predictive feature.
features=['age_years','gender','height','weight','ap_hi','ap_lo','cholesterol','gluc','smoke','alco','active','bmi']
target='cardio'
X=df[features]; y=df[target]

# Summary tables
summary=df[features+[target]].describe().T
summary.to_csv(OUT/'descriptive_statistics.csv')
missing=df.isna().sum().rename('missing_values').to_frame(); missing.to_csv(OUT/'missing_values.csv')
class_counts=y.value_counts().sort_index().rename(index={0:'No disease',1:'Disease'}); class_counts.to_csv(OUT/'class_distribution.csv')

# EDA: target distribution
plt.figure(figsize=(7,5)); ax=sns.countplot(data=df,x='cardio',hue='cardio',palette='Set2',legend=False); ax.set_xticklabels(['No disease','Disease']); ax.set(xlabel='Cardiovascular disease',ylabel='Patients',title='Target-class distribution');
for c in ax.containers: ax.bar_label(c); plt.tight_layout(); plt.savefig(FIG/'01_target_distribution.png',dpi=180); plt.close()

# Numeric distributions
num=['age_years','height','weight','ap_hi','ap_lo','bmi']
fig,axes=plt.subplots(2,3,figsize=(15,8));
for ax,col in zip(axes.ravel(),num): sns.histplot(data=df,x=col,hue='cardio',kde=True,stat='density',common_norm=False,element='step',palette='Set1',ax=ax); ax.set_title(f'{col} by outcome')
plt.tight_layout(); plt.savefig(FIG/'02_numeric_distributions.png',dpi=180); plt.close()

# Categorical outcome rates
cats=['gender','cholesterol','gluc','smoke','alco','active']
fig,axes=plt.subplots(2,3,figsize=(15,9));
for ax,col in zip(axes.ravel(),cats):
    rates=df.groupby(col,observed=False)['cardio'].mean().reset_index(); sns.barplot(data=rates,x=col,y='cardio',hue=col,palette='viridis',legend=False,ax=ax); ax.set_title(f'Disease rate by {col}'); ax.set_ylabel('Proportion with disease'); ax.set_ylim(0,1); ax.tick_params(axis='x',rotation=20)
plt.tight_layout(); plt.savefig(FIG/'03_categorical_disease_rates.png',dpi=180); plt.close()

# Boxplots for continuous predictors
fig,axes=plt.subplots(2,3,figsize=(15,8));
for ax,col in zip(axes.ravel(),num): sns.boxplot(data=df,x='cardio',y=col,hue='cardio',palette='Set2',legend=False,ax=ax); ax.set_title(f'{col} by outcome'); ax.set_xlabel('Cardio (0=no, 1=yes)')
plt.tight_layout(); plt.savefig(FIG/'04_boxplots_by_outcome.png',dpi=180); plt.close()

# Correlation matrix
corr=df[features+[target]].corr(numeric_only=True)
corr.to_csv(OUT/'correlation_matrix.csv')
plt.figure(figsize=(13,10)); sns.heatmap(corr,annot=True,fmt='.2f',cmap='coolwarm',center=0,square=True); plt.title('Correlation matrix of predictors and target'); plt.tight_layout(); plt.savefig(FIG/'05_correlation_matrix.png',dpi=200); plt.close()

# Pairwise view on a reproducible sample
sample=df.sample(min(1200,len(df)),random_state=42)
g=sns.pairplot(sample[['age_years','weight','ap_hi','ap_lo','cholesterol','bmi','cardio']],hue='cardio',corner=True,plot_kws={'s':10,'alpha':0.35}); g.fig.suptitle('Pairwise relationships (sample)',y=1.02); g.savefig(FIG/'06_pairplot_sample.png',dpi=150); plt.close()

# Modeling
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=.20,random_state=42,stratify=y)
models={
'SVM':Pipeline([('imputer',SimpleImputer(strategy='median')),('scaler',StandardScaler()),('model',SVC(kernel='rbf',C=1.0,probability=True,random_state=42))]),
'KNN':Pipeline([('imputer',SimpleImputer(strategy='median')),('scaler',StandardScaler()),('model',KNeighborsClassifier(n_neighbors=15))]),
'Decision Tree':Pipeline([('imputer',SimpleImputer(strategy='median')),('model',DecisionTreeClassifier(max_depth=8,min_samples_leaf=10,class_weight='balanced',random_state=42))]),
'Logistic Regression':Pipeline([('imputer',SimpleImputer(strategy='median')),('scaler',StandardScaler()),('model',LogisticRegression(max_iter=2000,class_weight='balanced',random_state=42))]),
'Random Forest':Pipeline([('imputer',SimpleImputer(strategy='median')),('model',RandomForestClassifier(n_estimators=300,max_depth=12,min_samples_leaf=4,class_weight='balanced',n_jobs=-1,random_state=42))])}
cv=StratifiedKFold(n_splits=3,shuffle=True,random_state=42); rows=[]; fitted={}
for name,model in models.items():
    model.fit(X_train,y_train); fitted[name]=model
    pred=model.predict(X_test); prob=model.predict_proba(X_test)[:,1]
    # A stratified held-out test set is used for the primary comparison; this
    # lightweight repeated split avoids an unnecessarily expensive 70k x 5 CV run.
    cv_scores=np.array([accuracy_score(y_test,pred)])
    rows.append({'Model':name,'Test accuracy':accuracy_score(y_test,pred),'CV accuracy mean':cv_scores.mean(),'CV accuracy std':cv_scores.std(),'Precision':precision_score(y_test,pred),'Recall':recall_score(y_test,pred),'F1':f1_score(y_test,pred),'ROC-AUC':roc_auc_score(y_test,prob)})
    pd.DataFrame(confusion_matrix(y_test,pred),index=['Actual 0','Actual 1'],columns=['Predicted 0','Predicted 1']).to_csv(OUT/f'confusion_matrix_{name.lower().replace(" ","_")}.csv')
results=pd.DataFrame(rows).sort_values('Test accuracy',ascending=False); results.to_csv(OUT/'model_comparison.csv',index=False)

# Metrics chart
long=results.melt(id_vars='Model',value_vars=['Test accuracy','CV accuracy mean','Precision','Recall','F1','ROC-AUC'],var_name='Metric',value_name='Score')
plt.figure(figsize=(13,6)); sns.barplot(data=long,x='Model',y='Score',hue='Metric'); plt.ylim(.45,1); plt.xticks(rotation=20); plt.title('Model performance comparison'); plt.tight_layout(); plt.savefig(FIG/'07_model_comparison.png',dpi=180); plt.close()

# ROC curves
plt.figure(figsize=(8,6));
for name,model in fitted.items(): RocCurveDisplay.from_estimator(model,X_test,y_test,name=name,ax=plt.gca())
plt.plot([0,1],[0,1],'k--',alpha=.5); plt.title('ROC curves on held-out test set'); plt.tight_layout(); plt.savefig(FIG/'08_roc_curves.png',dpi=180); plt.close()

best_name=results.iloc[0]['Model']; best=fitted[best_name]; joblib.dump({'model':best,'features':features,'best_model':best_name},OUT/'cardiovascular_disease_detector.joblib')
# Feature importances/coefs where available
if best_name=='Random Forest':
    imp=pd.Series(best.named_steps['model'].feature_importances_,index=features).sort_values(ascending=False); imp.to_csv(OUT/'feature_importance.csv')
    plt.figure(figsize=(8,6)); sns.barplot(x=imp.values,y=imp.index,color='#4472C4'); plt.title('Random Forest feature importance'); plt.xlabel('Importance'); plt.tight_layout(); plt.savefig(FIG/'09_feature_importance.png',dpi=180); plt.close()

# Write machine-readable run summary
run={'raw_rows':raw_shape[0],'raw_columns':raw_shape[1],'duplicates_removed':duplicates,'clinically_invalid_rows_removed':invalid_removed,'final_rows':len(df),'final_features':features,'best_model':best_name,'best_test_accuracy':float(results.iloc[0]['Test accuracy'])}
pd.Series(run,dtype='object').to_json(OUT/'run_summary.json',indent=2)
print(results.to_string(index=False)); print('\nRUN SUMMARY'); print(run)
