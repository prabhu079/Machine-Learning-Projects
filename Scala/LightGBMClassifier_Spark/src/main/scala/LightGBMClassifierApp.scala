import com.microsoft.ml.spark.lightgbm.{LightGBMClassifier, LightGBMConstants}
import org.apache.spark.ml.evaluation.BinaryClassificationEvaluator
import org.apache.spark.ml.feature.{StringIndexer, VectorAssembler}
import org.apache.spark.ml.{Pipeline, PipelineStage}
import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.types.{DoubleType, IntegerType, StructField, StructType}


object LightGBMClassifierApp extends App {
  val spark = SparkSession.builder().master("local[*]").appName("LightGBMClassifier").getOrCreate()

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
  val creditDf = spark.read.format("csv").option("header", true).option("mode", "DROPMALFORMED").schema(schema)
    .load("credit.csv").cache()
  val cols = Array("balance", "duration", "history", "purpose", "amount", "savings", "employment", "instPercent", "sexMarried",
    "guarantors", "residenceDuration", "assets", "age", "concCredit", "apartment", "credits", "occupation", "dependents", "hasPhone",
    "foreign")
  val assembler = new VectorAssembler().setInputCols(cols).setOutputCol("features").setHandleInvalid("skip")
  val featureDf = assembler.transform(creditDf)
  val indexer = new StringIndexer().setInputCol("creditability").setOutputCol("label")
  val seed = 5043
  val Array(trainingData, testData) = creditDf.randomSplit(Array(0.7, 0.3), seed)
  val lgbm = new LightGBMClassifier().setLabelCol("label").setLearningRate(0.1).setNumIterations(1000)
    .setObjective(LightGBMConstants.BinaryObjective).setFeaturesCol("features").setUseBarrierExecutionMode(true)
  val stages: Array[PipelineStage] = Array[PipelineStage](assembler, indexer, lgbm)
  val pipeline = new Pipeline().setStages(stages)
  val pModel = pipeline.fit(trainingData)
  val res = pModel.transform(testData)
  val evaluator = new BinaryClassificationEvaluator().setLabelCol("label")
    .setRawPredictionCol("rawPrediction")
  val accuracy = evaluator.evaluate(res)
  print("#############################################################################################" + accuracy)
}