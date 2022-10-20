# Databricks notebook source
# MAGIC %sql
# MAGIC 
# MAGIC select avg(radiant_win) from sandbox_apoiadores.abt_dota_pre_match

# COMMAND ----------

from sklearn import tree
from sklearn import ensemble
from sklearn import  metrics
import mlflow
from sklearn import model_selection

#import dos dados
sdf = spark.table("sandbox_apoiadores.abt_dota_pre_match")

df = sdf.toPandas()

# COMMAND ----------

target_column = 'radiant_win'
id_column = 'match_id'

features_columns = list(set(df.columns.tolist()) - set([target_column,id_column]))

y = df[target_column]
x = df[features_columns]

features_columns

# COMMAND ----------



x_train, x_test, y_train, y_test = model_selection.train_test_split(x,y,test_size=0.2,random_state=42)


# COMMAND ----------

print(x_train.shape[0])
print(x_test.shape[0])
print(y_train.shape[0])
print(y_test.shape[0])

# COMMAND ----------

mlflow.set_experiment("/Users/lg.stumpf@unesp.br/dota-unesp-LucasGaspar")

# COMMAND ----------

from sklearn.neural_network import MLPClassifier
with mlflow.start_run():

    #model = tree.DecisionTreeClassifier()
    mlflow.sklearn.autolog()
    #model = ensemble.AdaBoostClassifier(n_estimators=100 , learning_rate = 0.7)
    model = MLPClassifier()
    model.fit(x_train,y_train)
    
    y_train_pred = model.predict(x_train)
    y_train_prob = model.predict_proba(x_train)

    acc_train = metrics.accuracy_score(y_train, y_train_pred)
    
    y_test_pred = model.predict(x_test)
    y_test_prob = model.predict_proba(x_test)

    acc_test = metrics.accuracy_score(y_test, y_test_pred)

        

# COMMAND ----------

print(acc_train)
print(acc_test)
