import os

import numpy as np
from pyspark.ml import Pipeline
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.ml.feature import VectorAssembler, StringIndexer
from pyspark.sql.context import SQLContext
from pyspark.sql.functions import array, udf, to_timestamp, unix_timestamp, column
from pyspark.sql.session import SparkSession
from pyspark.sql.types import *

#from foo import foo


def main():
   #foo()
   os.putenv("hadoop.home.dir", "D:\\prabhu_python_Workspace\\Machine-Learning-Projects\\Cricket Match Prediction_SparkMlLib\\")
   spark = SparkSession.builder.master("local[*]").appName("CricketMatchPrediction").getOrCreate()
   sqlctx = SQLContext(spark)
   concat_udf = udf(lambda cols: "".join([x + " " for x in cols]).strip())
   df = sqlctx.read.load("train.csv", 'csv', header=True, inferSchema=True)
   df = df.withColumn('DateTime', concat_udf(array('DateOfGame', 'TimeOfGame')))
   df=df.withColumn('DateTime', unix_timestamp('DateTime', "dd-MM-yyyy HH:mm:ss"))
   df_test = sqlctx.read.load("test.csv", 'csv', header=True, inferSchema=True)
   df_test = df_test.withColumn('DateTime', concat_udf(array('DateOfGame', 'TimeOfGame'))) \
       .withColumn('DateTime', to_timestamp('DateTime', "dd-MM-yyyy HH:mm:ss"))
   df_test = df_test.withColumnRenamed("CityOfGame", "City")

   df_test = df_test.withColumn('DateTime', unix_timestamp(column('DateTime'), 'yyyy-MM-dd HH:mm:ss'))
   teams = set()
   cities = set()
   for row in (df.select('Team 1', 'Team 2', 'City').collect()):
       teams.add(row['Team 1'])
       teams.add(row['Team 2'])
       cities.add(row['City'])
   for row in (df_test.select('Team 1', 'Team 2', 'City').collect()):
       teams.add(row['Team 1'])
       teams.add(row['Team 2'])
       cities.add(row['City'])

   teams = np.sort(list(teams))
   cities = np.sort(list(cities))
   teamDict = {k: str(v + 1) for v, k in enumerate(teams)}
   cityDict = {k: str(v + 1) for v, k in enumerate(cities)}
   df = df.na.replace(teamDict, 1, 'Team 1').replace(teamDict, 1, 'Team 2').replace(cityDict, 1, 'City')
   df = df.withColumn('Team 1', df['Team 1'].cast(IntegerType())) \
       .withColumn('Team 2', df['Team 2'].cast(IntegerType())).withColumn('City', df['City'].cast(IntegerType()))
   df.dropna()
   df_test = df_test.na.replace(teamDict, 1, 'Team 1').replace(teamDict, 1, 'Team 2').replace(cityDict, 1, 'City')
   df_test = df_test.withColumn('Team 1', df_test['Team 1'].cast(IntegerType())) \
       .withColumn('Team 2', df_test['Team 2'].cast(IntegerType())).withColumn('City',
                                                                               df_test['City'].cast(IntegerType()))
   df_test.dropna()
   inputCols = df.columns[1:]
   inputCols.remove('DateOfGame')
   inputCols.remove('TimeOfGame')
   inputCols.remove('DayOfWeek')
   outputCol = 'Winner (team 1=1, team 2=0)'
   seed = 5043
   assembler = VectorAssembler(handleInvalid='skip').setInputCols(inputCols).setOutputCol('features')
   featureDf = assembler.transform(df.select(inputCols))
   feature_df_test = assembler.transform(df_test.select(inputCols))
   indexer = StringIndexer().setInputCol(outputCol).setOutputCol('label')
   labelModel = indexer.fit(featureDf)
   labelDf = labelModel.transform(featureDf)
   labelDf_test = labelModel.transform(feature_df_test)
   randomForestClassifier = RandomForestClassifier().setImpurity('gini').setMaxDepth(30).setNumTrees(200) \
       .setFeatureSubsetStrategy('auto').setSeed(seed)
   randomForestModel = randomForestClassifier.fit(labelDf)

   predictDf = randomForestModel.transform(labelDf_test)
   stages = [assembler, indexer, randomForestClassifier]
   pipeline = Pipeline().setStages(stages)
   pipelineModel = pipeline.fit(df.select(inputCols))
   pipelinePredictionDf = pipelineModel.transform(df_test.select(inputCols))
   evaluator = BinaryClassificationEvaluator().setLabelCol('label').setRawPredictionCol('rawPrediction')
   accuracy = evaluator.evaluate(predictDf)
   pipeLineAccuracy = evaluator.evaluate(pipelinePredictionDf)
   print("Accuracy on Test Data    : "+str(pipeLineAccuracy*100)+"%")

main()


