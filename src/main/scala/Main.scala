import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.evaluation.RegressionEvaluator
import org.apache.spark.ml.feature._
import org.apache.spark.ml.linalg.SparseVector
import org.apache.spark.ml.regression.{DecisionTreeRegressor, LinearRegression, RandomForestRegressor}
import org.apache.spark.ml.tuning.{CrossValidator, ParamGridBuilder}
import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.functions._
import org.apache.spark.sql.types.{IntegerType, StringType}


object Main {
  private val replaceNullWithUnknown = udf((x: String) => {
    var res = new String
    if (x == null || x == "Unknow" || x == "None" || x == "" || x == " ") { res = "unknown" }
    else { res = x }
    res
  }).asNondeterministic()

  private val replaceNAWithNull = udf((x: String) => {
    var res = new String
    if (x == "NA") { res = null }
    else { res = x }
    res
  }).asNondeterministic()

  private val replaceTimeWithDayPart = udf((x: Integer) => {
    var res = new String
    if (x >= 0 && x < 500) { res = "lateNight" }
    if (x >= 500 && x < 800) { res = "earlyMorning" }
    if (x >= 800 && x < 1200) { res = "lateMorning" }
    if (x >= 1200 && x < 1400) { res = "earlyAfternoon" }
    if (x >= 1400 && x < 1700) { res = "lateAfternoon" }
    if (x >= 1700 && x < 1900) { res = "earlyEvening" }
    if (x >= 1900 && x < 2100) { res = "lateEvening" }
    if (x >= 2100 && x <= 2400) { res = "earlyNight" }
    res
  }).asNondeterministic()


  def main(args: Array[String]): Unit = {
    val spark = SparkSession
      .builder()
      .appName("Big Data: Spark Practical Work")
      //.config("spark.master", "local")
      .master("local[12]")
      .getOrCreate()

    spark.sparkContext.setLogLevel("ERROR")

    val nDatasets = args.length
    var df = spark.emptyDataFrame

    if (nDatasets == 1) {
      val dataset = args(0)
      println()
      println("----------------------------------------------------- DATASET SELECTED: " + dataset + " -----------------------------------------------------")
      println()
      df = spark.read.option("header", value = "true").csv("hdfs://localhost:9000/datasets/" + dataset + ".csv")
    }
    else {
      println()
      println("------------------------------------------------------------ DATASETS: -----------------------------------------------------------")
      println()
      var dataset = args(0)
      df = spark.read.option("header", value = "true").csv("hdfs://localhost:9000/datasets/" + dataset + ".csv")
      println()
      println("-------------------------------------------------------- DATASET 1: " + dataset + " ---------------------------------------------------------")
      println()
      for (i <- 1 until nDatasets) {
        dataset = args(i)
        println()
        println("-------------------------------------------------------- DATASET " + (i+1) + ": " + dataset + " ---------------------------------------------------------")
        println()
        df = spark.read.option("header", value = "true").csv("hdfs://localhost:9000/datasets/" + dataset + ".csv")
      }
    }


    println()
    println("----------------------------------------------------------------------------------------------------------------------------------")
    println("--------------------------------------------------- DATA LOADING -----------------------------------------------------------------")
    println("----------------------------------------------------------------------------------------------------------------------------------")
    println()

    var dfPlane = spark.read.option("header", value = "true").csv("src/main/resources/plane-data.csv")


    println("----------------------------------------------------------------------------------------------------------------------------------")
    println("----------------------------------------- DATA PREPROCESSING & FEATURE SELECTION -------------------------------------------------")
    println("----------------------------------------------------------------------------------------------------------------------------------")
    println()

    // We delete the forbidden columns
    println("--------------------------------- We delete the forbidden columns ----------------------------------")
    val columnsToDrop = Array("ArrTime", "ActualElapsedTime", "AirTime", "TaxiIn", "Diverted", "CarrierDelay", "WeatherDelay", "NASDelay", "SecurityDelay", "LateAircraftDelay")
    df = df.drop(columnsToDrop: _*)
    println("----------------------------------------------- Done -----------------------------------------------")
    println()


    // We delete the null values of the target variable as we are not going to use them
    println("----------------------------- We delete the null values of \"ArrDelay\" ------------------------------")
    df = df.filter("ArrDelay is NOT NULL AND ArrDelay NOT LIKE 'NA'")
    println("----------------------------------------------- Done -----------------------------------------------")
    println()
    println("Target variable: ArrDelay")
    println()


    // We delete all the rows that contain cancelled flights, since this will not be useful for our prediction goal
    println("----------------------- We delete all the rows that contain cancelled flights ----------------------")
    df = df.filter("Cancelled == 0")
    println("----------------------------------------------- Done -----------------------------------------------")
    println()

    // Therefore, we eliminate the "CancellationCode" and "Cancelled" columns
    println("-------------------- We eliminate the \"CancellationCode\" and \"Cancelled\" columns -------------------")
    df = df.drop("Cancelled", "CancellationCode")
    println("----------------------------------------------- Done -----------------------------------------------")
    println()


    // We delete the "year" and "status" columns since they do not provide more useful information
    println("------------------------- We delete \"year\" and \"status\" in dfPlane dataset -------------------------")
    dfPlane = dfPlane.drop("year").drop("status")
    println("----------------------------------------------- Done -----------------------------------------------")
    println()


    // We join the datasets (the dataset/s and the plane-data dataset) if the matching variable (TailNum) is not null
    var included = false
    if (df.groupBy("tailNum").count().groupBy("tailNum").count().count() > 1) {
      println("--------------------------------------- Joining the datasets ---------------------------------------")
      included = true
      df = df.join(dfPlane, "tailNum")
      println("----------------------------------------------- Done -----------------------------------------------")
      println()
    }

    df.show()

    // We delete the "TailNum", "UniqueCarrier" and "FlightNum" columns as they are IDs and do not provide huge value
    println("----------------- We delete the \"TailNum\", \"UniqueCarrier\" and \"FlightNum\" columns -----------------")
    df = df.drop("TailNum", "UniqueCarrier", "FlightNum")
    println("----------------------------------------------- Done -----------------------------------------------")
    println()


    // If the plane-data dataset is included, we clean the "issue_date" column as it is going to be used later
    if (included) {
      println("--------------------------------------- We clean \"issue_date\" --------------------------------------")
      df = df.filter("issue_date is NOT NULL AND issue_date NOT LIKE 'None' AND issue_date NOT LIKE 'NA'")
      println("----------------------------------------------- Done -----------------------------------------------")
      println()
    }


    // If the plane-data dataset is included, we delete the plane tailnumbers that do not have any data
    if (included) {
      println("-------- We delete the plane tailnumbers that do not have any data from plane-data dataset ---------")
      df = df.filter("type is NOT NULL AND manufacturer is NOT NULL AND model is NOT NULL AND aircraft_type is NOT NULL AND engine_type is NOT NULL")
      println("----------------------------------------------- Done -----------------------------------------------")
      println()
    }


    // We check for NA values in each column of the dataset and set them to null for the imputers to do their work
    println("-------------------- Checking for NA values in the dataset to set them to null ---------------------")
    for (i <- 0 until df.columns.drop(df.columns.indexOf("ArrDelay")).length) {
      val column = df.columns(i)
      df = df.withColumn(column, replaceNAWithNull(col(column)))
    }
    println("----------------------------------------------- Done -----------------------------------------------")
    println()


    println("------------------------- We delete the columns that only have NULL values -------------------------")
    // Numerical columns for "mean" imputer and "most frequent" imputer
    // Categorical columns for the One Hot Encoder
    var numColsMean = Array("DepTime", "CRSDepTime", "CRSArrTime", "CRSElapsedTime", "DepDelay", "Distance", "TaxiOut")
    var numColsMf = Array("Year", "Month", "DayofMonth", "DayOfWeek")
    var catColsDf = Array("Origin", "Dest")
    var columnsToDrop2 = df.columns

    if (included) {
      catColsDf = catColsDf ++ Array("type", "manufacturer", "model", "aircraft_type", "engine_type")
    }

    for (i <- 0 until df.columns.length) {
      val column = df.columns(i)
      if (column == "Year" || df.groupBy(column).count().groupBy(column).count().count() > 1) {
        columnsToDrop2 = columnsToDrop2.filter(_ != column)
      }
      else {
        if (numColsMean.contains(column)) {
          numColsMean = numColsMean.filter(_ != column)
        }
        else if (numColsMf.contains(column)) {
          numColsMf = numColsMf.filter(_ != column)
        }
        else if (catColsDf.contains(column)) {
          catColsDf = catColsDf.filter(_ != column)
        }
      }
    }

    df = df.drop(columnsToDrop2: _*).cache()
    var numCols = numColsMean ++ numColsMf
    println("----------------------------------------------- Done -----------------------------------------------")
    println()

    df.show()

    // We cast to Integer every column in order to be able to use the imputer
    println("-------------- We cast to Integer every column in order to be able to use the imputer --------------")
    for (i <- 0 until df.columns.length) {
      val colName = df.columns(i)
      if (numCols.contains(colName) || colName == "ArrDelay") {
        df = df.withColumn(colName, col(colName).cast(IntegerType))
      }
    }
    println("----------------------------------------------- Done -----------------------------------------------")
    println()


    // We look at the correlation between the target variable and the explanatory variables
    println("------------------ Correlations between explanatory variables and target variable ------------------")
    for (i <- 0 until numCols.length) {
      val column = numCols(i)
      println("Correlation between ArrDelay and " + column + ":")
      println(df.stat.corr("ArrDelay", column, "pearson"))
    }
    println("----------------------------------------------- Done -----------------------------------------------")
    println()

    // Moreover, we also look at the correlation between the explanatory variables. If any are high correlated
    // that indicates that one of them could be removed, as they produce a similar effect on the target variable
    println("---------------------------- Correlations between explanatory variables ----------------------------")
    for (i <- 0 until numCols.length) {
      val column = numCols(i)
      for (j <- i + 1 until numCols.length) {
        val column2 = numCols(j)
        println("Correlation between " + column + " and " + column2 + ":")
        println(df.stat.corr(column, column2, "pearson"))
      }
    }
    println("----------------------------------------------- Done -----------------------------------------------")
    println()


    // We delete the "CRSDepTime" and "CRSElapsedTime" columns as the correlation tell us that they produce a similar effect on the target variable
    println("--------------------- We delete the \"CRSDepTime\" and \"CRSElapsedTime\" columns ----------------------")
    df = df.drop("CRSDepTime", "CRSElapsedTime")
    numCols = numCols.filter(_ != "CRSDepTime").filter(_ != "CRSElapsedTime")
    numColsMean = numColsMean.filter(_ != "CRSDepTime").filter(_ != "CRSElapsedTime")
    println("----------------------------------------------- Done -----------------------------------------------")
    println()


    // We apply the "most frequent" imputer for some numerical columns
    println("-------------------------------- We apply the \"most frequent\" imputer ------------------------------")
    val imputer = new Imputer()
      .setInputCols(numColsMf)
      .setOutputCols(numColsMf)
      .setStrategy("mode")
    df = imputer.fit(df).transform(df)
    println("----------------------------------------------- Done -----------------------------------------------")
    println()


    // We apply the "mean" imputer for the rest of the numerical columns
    println("----------------------------------- We apply the \"mean\" imputer ------------------------------------")
    imputer.setInputCols(numColsMean).setOutputCols(numColsMean).setStrategy("mean")
    df = imputer.fit(df).transform(df)
    println("----------------------------------------------- Done -----------------------------------------------")
    println()


    // If the plane-data dataset is included, we create the column "PlaneAge" from the data in "Year" and "issue_date"
    if (included) {
      println("---------------- We create the column \"PlaneAge\" and remove the column \"issue_date\" ----------------")
      df = df.withColumnRenamed("issue_date", "PlaneAge")
      df = df.withColumn("PlaneAge", col("Year") - year(to_date(col("PlaneAge").cast(StringType), "M/d/y")))
      df = df.withColumn("PlaneAge", when(col("PlaneAge") < 0, 0).otherwise(col("PlaneAge")))
      numCols = numCols ++ Array("PlaneAge")
      println("----------------------------------------------- Done -----------------------------------------------")
      println()
    }


    // We check for null values in the categorical columns and swap them with "unknown"
    println("--------- We check for null values in the categorical columns and swap them with \"unknown\" ---------")
    for (i <- 0 until catColsDf.length) {
      val column = catColsDf(i)
      df = df.withColumn(column, replaceNullWithUnknown(col(column)))
    }
    println("----------------------------------------------- Done -----------------------------------------------")
    println()

    // We specify that the values for "DepTime" and "CRSArrTime" columns should be hour-like
    println("------- We specify that the values for \"DepTime\" and \"CRSArrTime\" columns should be hour-like ------")
    df = df.filter("DepTime <= 2400").filter("CRSArrTime <= 2400")
    println("----------------------------------------------- Done -----------------------------------------------")
    println()


    // We change the value of "DepTime" and "CRSArrTime" to strings containing values such as morning, night... in order to apply one hot encoder more efficiently
    println("----------------------- We change the value of \"DepTime\" and \"CRSArrTime\" --------------------------")
    df = df.withColumn("DepTime", replaceTimeWithDayPart(col("DepTime")))
    df = df.withColumn("CRSArrTime", replaceTimeWithDayPart(col("CRSArrTime")))
    numCols = numCols.filter(_ != "DepTime").filter(_ != "CRSDepTime").filter(_ != "CRSArrTime")
    println("----------------------------------------------- Done -----------------------------------------------")
    println()

    df.show()


    // We divide the variables into numerical/continuous and categorical
    var columnsToIndex = Array[String]()
    var catCols = Array[String]()
    var indexedColumns = Array[String]()

    for (i <- 0 until df.columns.length) {
      val column = df.columns(i)
      if (!numCols.contains(column) && column != "ArrDelay") {
        columnsToIndex = columnsToIndex ++ Array(column)
        catCols = catCols ++ Array(column.concat("Cat"))
        indexedColumns = indexedColumns ++ Array(column.concat("Indexed"))
      }
    }


    // Declaration of the indexer that will transform entries to integer values
    println("------------- Declaration of the indexer that will transform entries to integer values -------------")
    val indexer = new StringIndexer()
      .setInputCols(columnsToIndex)
      .setOutputCols(indexedColumns)
    println("----------------------------------------------- Done -----------------------------------------------")
    println()


    // Declaration of the one hot encoder that will process the categorical variables
    println("---------- Declaration of the one hot encoder that will process the categorical variables ----------")
    val ohe = new OneHotEncoder()
      .setInputCols(indexedColumns)
      .setOutputCols(catCols)
    println("----------------------------------------------- Done -----------------------------------------------")
    println()

    val assCols = numCols ++ catCols

    // Declaration of the assembler that will extract the features from our variables
    println("-------------------------------- Extracting features from our data ---------------------------------")
    val assembler = new VectorAssembler()
      .setInputCols(assCols)
      .setOutputCol("features")
    println("----------------------------------------------- Done -----------------------------------------------")
    println()


    // Normalizing the extracted features
    println("-------------------------------- Normalizing the extracted features --------------------------------")
    val normalizer = new Normalizer()
      .setInputCol("features")
      .setOutputCol("normFeatures")
      .setP(1.0)
    println("----------------------------------------------- Done -----------------------------------------------")
    println()


    // We use a pipeline in order to create a sequence of run stages
    println("---------------------------------------- Use of a pipeline -----------------------------------------")
    val pipeline = new Pipeline()
      .setStages(Array(indexer, ohe, assembler, normalizer))
    df = pipeline.fit(df).transform(df)
    df.printSchema()
    println("----------------------------------------------- Done -----------------------------------------------")
    println()


    df = df.drop(indexedColumns: _*)
    df = df.drop(columnsToIndex: _*)
    df = df.drop(catCols: _*)
    df = df.drop(Array("Year", "DayOfWeek", "DepDelay", "Distance", "TaxiOut", "DepTime", "CSRArrTime", "Month", "DayofMonth", "features"): _*)
    if (included) {
      df = df.drop(Array("PlaneAge"): _*)
    }
    df.show()


    println("----------------------------------------------------------------------------------------------------------------------------------")
    println("--------------------------------------------------------- DATA MODELING ----------------------------------------------------------")
    println("----------------------------------------------------------------------------------------------------------------------------------")
    println()

    println("--------------------------------------------------- FEATURE STEPWISE SELECTION ---------------------------------------------------")
    println()

    val selectorFalseDiscoveryRate = new UnivariateFeatureSelector()
      .setFeatureType("continuous")
      .setLabelType("continuous")
      .setSelectionMode("fdr")
      .setSelectionThreshold(0.05)
      .setFeaturesCol("normFeatures")
      .setLabelCol("ArrDelay")
      .setOutputCol("selectedFeatures")


    val selectorFamilywiseErrorRate = new UnivariateFeatureSelector()
      .setFeatureType("continuous")
      .setLabelType("continuous")
      .setSelectionMode("fwe")
      .setSelectionThreshold(0.05)
      .setFeaturesCol("normFeatures")
      .setLabelCol("ArrDelay")
      .setOutputCol("selectedFeatures")

    val row = df.select("normFeatures").head
    val vector = row(0).asInstanceOf[SparseVector]
    println("------------------------------ Number of features: " + vector.size + " ------------------------------")
    println()

    val fdr = selectorFalseDiscoveryRate.fit(df)
    val dfFdr = fdr.transform(df)
    println("---------------------- Number of features after using FDR: " + fdr.selectedFeatures.length + " ----------------------")
    println()

    val fwe = selectorFamilywiseErrorRate.fit(df)
    val dfFwe = fwe.transform(df)
    println("---------------------- Number of features after using FWE: " + fwe.selectedFeatures.length + " ----------------------")
    println()

    val Array(trainingDataFdr, testDataFdr) = dfFdr.randomSplit(Array(0.7, 0.3), 10)
    val Array(trainingDataFwe, testDataFwe) = dfFwe.randomSplit(Array(0.7, 0.3), 10)


    println("------------------------------------------------------- LINEAR REGRESSION --------------------------------------------------------")

    // We create a linear regression learning algorithm
    val linearRegression = new LinearRegression()
      .setLabelCol("ArrDelay")
      .setFeaturesCol("normFeatures")
      .setPredictionCol("predictionLR")


    // We define a grid of hyper-parameters values to search over
    val lrParamGrid = new ParamGridBuilder()
      .addGrid(linearRegression.regParam, Array(0.01))
      .addGrid(linearRegression.elasticNetParam, Array(0.25))
      .addGrid(linearRegression.maxIter, Array(10))
      .build()


    // We create a regression evaluator for using the Root Mean Squared Error metric
    val lrEvaluatorRMSE = new RegressionEvaluator()
      .setLabelCol("ArrDelay")
      .setPredictionCol("predictionLR")
      .setMetricName("rmse")


    // We create a regression evaluator for using the R Squared metric
    val lrEvaluatorR2 = new RegressionEvaluator()
      .setLabelCol("ArrDelay")
      .setPredictionCol("predictionLR")
      .setMetricName("r2")


    // We define a 5-fold cross-validator
    val lrCrossValidator = new CrossValidator()
      .setEstimator(linearRegression)
      .setEvaluator(lrEvaluatorRMSE)
      .setEstimatorParamMaps(lrParamGrid)
      .setNumFolds(5)


    // We train and tune the model using k-fold cross validation
    // to then use the best model to make predictions on the test data
    // to evaluate the predictions using the chosen evaluation metric
    val lrModelFdr = lrCrossValidator.fit(trainingDataFdr)
    println("Model parameters in LR - False Discovery Rate:")
    println(lrModelFdr.bestModel.extractParamMap())

    val lrPredictionsFdr = lrModelFdr.transform(testDataFdr)
    println("ArrDelay VS predictionLR - False Discovery Rate:")
    lrPredictionsFdr.select("ArrDelay", "predictionLR").show(10, truncate = false)

    println("------------------------ LR: Root Mean Squared Error - False Discovery Rate ------------------------")
    println(lrEvaluatorRMSE.evaluate(lrPredictionsFdr))
    println("------------------- LR: Coefficient of Determination (R2) - False Discovery Rate -------------------")
    println(lrEvaluatorR2.evaluate(lrPredictionsFdr))
    println()


    val lrModelFwe = lrCrossValidator.fit(trainingDataFwe)
    println("Model parameters in LR - Family-wise Error Rate:")
    println(lrModelFwe.bestModel.extractParamMap())

    val lrPredictionsFwe = lrModelFwe.transform(testDataFwe)
    println("ArrDelay VS predictionLR - Family-wise Error Rate:")
    lrPredictionsFwe.select("ArrDelay", "predictionLR").show(10, truncate = false)

    println("----------------------- LR: Root Mean Squared Error - Family-wise Error Rate -----------------------")
    println(lrEvaluatorRMSE.evaluate(lrPredictionsFwe))
    println("------------------ LR: Coefficient of Determination (R2) - Family-wise Error Rate ------------------")
    println(lrEvaluatorR2.evaluate(lrPredictionsFwe))
    println()


    println("---------------------------------------------------- DECISION TREE REGRESSOR -----------------------------------------------------")

    // We create a decision tree regressor algorithm
    val decisionTree = new DecisionTreeRegressor()
      .setLabelCol("ArrDelay")
      .setFeaturesCol("normFeatures")
      .setPredictionCol("predictionDTR")


    // We create a regression evaluator for using the Root Mean Squared Error metric
    val dtrEvaluatorRMSE = new RegressionEvaluator()
      .setLabelCol("ArrDelay")
      .setPredictionCol("predictionDTR")
      .setMetricName("rmse")


    // We create a regression evaluator for using the R Squared metric
    val dtrEvaluatorR2 = new RegressionEvaluator()
      .setLabelCol("ArrDelay")
      .setPredictionCol("predictionDTR")
      .setMetricName("r2")


    // We define a 5-fold cross-validator
    val dtrCrossValidator = new CrossValidator()
      .setEstimator(decisionTree)
      .setEvaluator(dtrEvaluatorRMSE)
      .setEstimatorParamMaps(new ParamGridBuilder().build())
      .setNumFolds(5)


    // We train and tune the model using k-fold cross validation
    // to then use the best model to make predictions on the test data
    // to evaluate the predictions using the chosen evaluation metric
    val dtrModelFdr = dtrCrossValidator.fit(trainingDataFdr)
    println("Model parameters in DTR - False Discovery Rate:")
    println(dtrModelFdr.bestModel.extractParamMap())

    val dtrPredictionsFdr = dtrModelFdr.transform(testDataFdr)
    println("ArrDelay VS predictionDTR - False Discovery Rate:")
    dtrPredictionsFdr.select("ArrDelay", "predictionDTR").show(10, truncate = false)

    println("------------------------ DTR: Root Mean Squared Error - False Discovery Rate ------------------------")
    println(dtrEvaluatorRMSE.evaluate(dtrPredictionsFdr))
    println("------------------- DTR: Coefficient of Determination (R2) - False Discovery Rate -------------------")
    println(dtrEvaluatorR2.evaluate(dtrPredictionsFdr))
    println()


    val dtrModelFwe = dtrCrossValidator.fit(trainingDataFwe)
    println("Model parameters in DTR - Family-wise Error Rate:")
    println(dtrModelFwe.bestModel.extractParamMap())

    val dtrPredictionsFwe = dtrModelFwe.transform(testDataFwe)
    println("ArrDelay VS predictionDTR - Family-wise Error Rate:")
    dtrPredictionsFwe.select("ArrDelay", "predictionDTR").show(10, truncate = false)

    println("----------------------- DTR: Root Mean Squared Error - Family-wise Error Rate -----------------------")
    println(dtrEvaluatorRMSE.evaluate(dtrPredictionsFwe))
    println("------------------ DTR: Coefficient of Determination (R2) - Family-wise Error Rate ------------------")
    println(dtrEvaluatorR2.evaluate(dtrPredictionsFwe))
    println()


    println("----------------------------------------------------- RANDOM FOREST REGRESSOR ----------------------------------------------------")

    // We create a random forest regressor algorithm
    val randomForest = new RandomForestRegressor()
      .setLabelCol("ArrDelay")
      .setFeaturesCol("normFeatures")
      .setPredictionCol("predictionRF")


    // We create a regression evaluator for using the R Squared metric
    val rfEvaluatorR2 = new RegressionEvaluator()
      .setLabelCol("ArrDelay")
      .setPredictionCol("predictionRF")
      .setMetricName("r2")


    // We create a regression evaluator for using the Root Mean Squared Error metric
    val rfEvaluatorRMSE = new RegressionEvaluator()
      .setLabelCol("ArrDelay")
      .setPredictionCol("predictionRF")
      .setMetricName("rmse")


    // We define a 5-fold cross-validator
    val rfCrossValidator = new CrossValidator()
      .setEstimator(randomForest)
      .setEvaluator(rfEvaluatorRMSE)
      .setEstimatorParamMaps(new ParamGridBuilder().build())
      .setNumFolds(5)


    // We train and tune the model using k-fold cross validation
    // to then use the best model to make predictions on the test data
    // to evaluate the predictions using the chosen evaluation metric
    val rfModelFdr = rfCrossValidator.fit(trainingDataFdr)
    println("Model parameters in RF - False Discovery Rate:")
    println(rfModelFdr.bestModel.extractParamMap())

    val rfPredictionsFdr = rfModelFdr.transform(testDataFdr)
    println("ArrDelay VS predictionRF - False Discovery Rate:")
    rfPredictionsFdr.select("ArrDelay", "predictionRF").show(10, truncate = false)

    println("------------------------ RF: Root Mean Squared Error - False Discovery Rate ------------------------")
    println(rfEvaluatorRMSE.evaluate(rfPredictionsFdr))
    println("------------------- RF: Coefficient of Determination (R2) - False Discovery Rate -------------------")
    println(rfEvaluatorR2.evaluate(rfPredictionsFdr))
    println()


    val rfModelFwe = rfCrossValidator.fit(trainingDataFwe)
    println("Model parameters in RF - Family-wise Error Rate:")
    println(rfModelFwe.bestModel.extractParamMap())

    val rfPredictionsFwe = rfModelFwe.transform(testDataFwe)
    println("ArrDelay VS predictionRF - Family-wise Error Rate:")
    rfPredictionsFwe.select("ArrDelay", "predictionRF").show(10, truncate = false)

    println("----------------------- RF: Root Mean Squared Error - Family-wise Error Rate -----------------------")
    println(rfEvaluatorRMSE.evaluate(rfPredictionsFwe))
    println("------------------ RF: Coefficient of Determination (R2) - Family-wise Error Rate ------------------")
    println(rfEvaluatorR2.evaluate(rfPredictionsFwe))
    println()


    // We create a summary of the RMSE and R2 measures to see which model performs better
    // R2 explains the variability of the target variable that is explained by the explanatory variables
    // RMSE measures the difference between the predicted values and the actual ones
    println("----------------------------------------------- SUMMARY OF THE MODELS' PERFORMANCE -----------------------------------------------")
    println()

    println("------------------------------------ LINEAR REGRESSION with FDR ------------------------------------")
    println(lrEvaluatorRMSE.evaluate(lrPredictionsFdr))
    println(lrEvaluatorR2.evaluate(lrPredictionsFdr))
    println("------------------------------------ LINEAR REGRESSION with FWE ------------------------------------")
    println(lrEvaluatorRMSE.evaluate(lrPredictionsFwe))
    println(lrEvaluatorR2.evaluate(lrPredictionsFwe))
    println()

    println("--------------------------------- DECISION TREE REGRESSOR with FDR ---------------------------------")
    println(dtrEvaluatorRMSE.evaluate(dtrPredictionsFdr))
    println(dtrEvaluatorR2.evaluate(dtrPredictionsFdr))
    println("--------------------------------- DECISION TREE REGRESSOR with FWE ---------------------------------")
    println(dtrEvaluatorRMSE.evaluate(dtrPredictionsFwe))
    println(dtrEvaluatorR2.evaluate(dtrPredictionsFwe))
    println()

    println("--------------------------------- RANDOM FOREST REGRESSOR with FDR ---------------------------------")
    println(rfEvaluatorRMSE.evaluate(rfPredictionsFdr))
    println(rfEvaluatorR2.evaluate(rfPredictionsFdr))
    println("--------------------------------- RANDOM FOREST REGRESSOR with FWE ---------------------------------")
    println(rfEvaluatorRMSE.evaluate(rfPredictionsFwe))
    println(rfEvaluatorR2.evaluate(rfPredictionsFwe))

  }
}