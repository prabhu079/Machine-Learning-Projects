name := "LightGBMClassifier_Spark"

version := "0.1"

scalaVersion := "2.11.7"

libraryDependencies ++= {
  Seq(
    "org.apache.spark" %% "spark-core" % "2.4.4",
    "org.apache.spark" %% "spark-sql" % "2.4.4",
    "org.apache.spark" %% "spark-mllib" % "2.4.4",
    "org.slf4j" % "slf4j-api" % "1.7.5",
    "ch.qos.logback" % "logback-classic" % "1.0.9",
    "org.apache.hadoop" % "hadoop-common" % "2.7.1",
    "com.microsoft.ml.spark" %% "mmlspark" % "0.18.1"
  )
}

resolvers ++= Seq(
  "Typesafe repository" at "http://repo.typesafe.com/typesafe/releases/",
  "mmlLib repository" at "https://mvnrepository.com/artifact/com.microsoft.ml.spark/mmlspark",
  "winutils" at "https://mvnrepository.com/artifact/org.apache.hadoop/hadoop-winutils"
)