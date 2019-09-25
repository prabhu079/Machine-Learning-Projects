import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.classification.RandomForestClassifier
import org.apache.spark.ml.evaluation.BinaryClassificationEvaluator
import org.apache.spark.ml.feature.{StringIndexer, VectorAssembler}
import org.apache.spark.ml.tuning.{CrossValidator, CrossValidatorModel, ParamGridBuilder}
import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.types.{DoubleType, IntegerType, StructField, StructType}

object RandomForestLab extends App {
  val spark = SparkSession.builder().master("local[*]").appName("CreditRiskClassifier").getOrCreate()

  import spark.implicits._

  val schema = StructType(Seq[StructField](StructField("creditability", DoubleType, nullable = true),
    StructField("balance", DoubleType, nullable = true),
    StructField("duration", IntegerType, nullable = true),
    StructField("history", IntegerType, nullable = true),
    StructField("purpose", IntegerType, nullable = true),
    StructField("amount", IntegerType, nullable = true),
    StructField("savings", IntegerType, nullable = true),
    StructField("employment", IntegerType, nullable = true),
    StructField("instPercent", IntegerType, nullable = true),
    StructField("sexMarried", IntegerType, nullable = true),
    StructField("guarantors", IntegerType, nullable = true),
    StructField("residenceDuration", IntegerType, nullable = true),
    StructField("assets", IntegerType, nullable = true),
    StructField("age", IntegerType, nullable = true),
    StructField("concCredit", IntegerType, nullable = true),
    StructField("apartment", IntegerType, nullable = true),
    StructField("credits", IntegerType, nullable = true),
    StructField("occupation", IntegerType, nullable = true),
    StructField("dependents", IntegerType, nullable = true),
    StructField("hasPhone", IntegerType, nullable = true),
    StructField("foreign", IntegerType, nullable = true)
  ))

  val creditDf = spark.read.format("csv").option("header", true)
    .option("mode", "DROPMALFORMED")
    .schema(schema)
    .load("credit.csv").cache()
  creditDf.printSchema()
  val cols = Array("balance", "duration", "history", "purpose", "amount", "savings", "employment", "instPercent", "sexMarried",
    "guarantors", "residenceDuration", "assets", "age", "concCredit", "apartment", "credits", "occupation", "dependents", "hasPhone",
    "foreign")
  val assembler = new VectorAssembler().setInputCols(cols).setOutputCol("features")
  val featureDf = assembler.transform(creditDf)
  val indexer = new StringIndexer().setInputCol("creditability").setOutputCol("label")
  val labelDf = indexer.fit(featureDf).transform(featureDf)
  labelDf.show(10)
  val seed = 5043
  val Array(trainingData, testData) = labelDf.randomSplit(Array(0.7, 0.3), seed)
  val randomForestClassifier = new RandomForestClassifier().setImpurity("gini")
    .setMaxDepth(20).setNumTrees(20).setFeatureSubsetStrategy("auto").setSeed(seed)
  val randomForestModel=randomForestClassifier.fit(trainingData)
  val predictDf=randomForestModel.transform(testData)
  val Array(pipeLineTrainingData,pipeLineTestData)=creditDf.randomSplit(Array(0.7,0.3),seed)
  val stages=Array(assembler,indexer,randomForestClassifier)
  val pipeline=new Pipeline().setStages(stages)
  val pipelineModel=pipeline.fit(pipeLineTrainingData)
  val pipelinePredictDf=pipelineModel.transform(pipeLineTestData)
  val evaluator=new BinaryClassificationEvaluator()
    .setLabelCol("label")
    .setMetricName("areaUnderROC")
    .setRawPredictionCol("rawPrediction")
  val accuracy=evaluator.evaluate(predictDf)
  val pipelineAccuracy=evaluator.evaluate(pipelinePredictDf)
  println(accuracy)
  println(pipelineAccuracy)

  val paramGrid = new ParamGridBuilder()
    .addGrid(randomForestClassifier.maxBins, Array(25, 28, 31))
    .addGrid(randomForestClassifier.maxDepth, Array(4, 6, 8))
    .addGrid(randomForestClassifier.impurity, Array("entropy", "gini"))
    .build()

  // define cross validation stage to search through the parameters
  // K-Fold cross validation with BinaryClassificationEvaluator
  val cv = new CrossValidator()
    .setEstimator(pipeline)
    .setEvaluator(evaluator)
    .setEstimatorParamMaps(paramGrid)
    .setNumFolds(5)
  val cvModel = cv.fit(pipeLineTrainingData)

  // test cross validated model with test data
  val cvPredictionDf = cvModel.transform(pipeLineTestData)
  cvPredictionDf.show(10)

  // measure the accuracy of cross validated model
  // this model is more accurate than the old model
  val cvAccuracy = evaluator.evaluate(cvPredictionDf)
  println(cvAccuracy)
  /*cvModel.write.overwrite()
    .save("./models/credit-model")
*/
  // load CrossValidatorModel model here
  /*val cvModelLoaded = CrossValidatorModel
    .load("./models/credit-model")*/





}


