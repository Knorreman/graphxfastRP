import fastp.FastRP.fastRP
import org.apache.spark.{SparkConf, SparkContext}
import org.apache.spark.graphx.{Edge, EdgeContext, EdgeDirection, EdgeTriplet, Graph, PartitionStrategy, VertexId, VertexRDD}
import org.apache.spark.mllib.classification.{LogisticRegressionWithLBFGS, NaiveBayes, SVMModel, SVMWithSGD}
import org.apache.spark.mllib.evaluation.{BinaryClassificationMetrics, MulticlassMetrics}
import org.apache.spark.mllib.feature.StandardScaler
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.mllib.optimization.SquaredL2Updater
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.mllib.tree.{DecisionTree, RandomForest}
import org.apache.spark.rdd.RDD
import org.apache.spark.storage.StorageLevel

import scala.collection.mutable.ArrayBuffer
import scala.util.Random
object Main {
  def main(args: Array[String]): Unit = {
    println("Hello world!")
    val conf = new SparkConf()
      .setAppName("FastRP")
      .setMaster("local[*]")
      .set("spark.driver.memory", "32g")
      .set("spark.rdd.compress", "true")
      .set("spark.serializer", "org.apache.spark.serializer.KryoSerializer")

    val sc = new SparkContext(conf)
    sc.setLogLevel("WARN")

    val tw_graph = load_twitch_edges(sc)
      .partitionBy(PartitionStrategy.RandomVertexCut)

    tw_graph.vertices.take(10).foreach(println)
    println("num vertices:")
    println(tw_graph.vertices.count())
    println("num edges:")
    println(tw_graph.edges.count())

    val eigenCentrality = tw_graph.staticPageRank(5, 0.0)

    eigenCentrality.vertices
      .sortBy(v => v._2, ascending = false)
      .take(25)
      .foreach(println)

    val tw_features = load_twitch_features(sc)
    tw_features.take(10).foreach(println)

    tw_features.keyBy(t => t.language)
      .mapValues(_ => 1.0)
      .reduceByKey(_ + _)
      .collect()
      .foreach(println)

    println("English or not:")
    tw_features.keyBy(t => t.englishOrNot)
      .mapValues(_ => 1.0)
      .reduceByKey(_ + _)
      .collect()
      .foreach(println)
//    val graph = GraphGenerators.logNormalGraph(sc, 4096).mapVertices((_, i) => i.toLong)
//      .partitionBy(PartitionStrategy.EdgePartition2D)
//
//    val rpGraph = fastRP(graph, 8, 10)
//
//    rpGraph.vertices
//      .mapValues(x => x.toArray.mkString("Array(", ", ", ")"))
//      .take(20)
//      .foreach(println)

    val weights = Array(0.0f, 0.0f, 1.0f, 1.0f)
    val twRPGraph = fastRP(tw_graph, 32, weights)

    twRPGraph.vertices
      .mapValues(x => x.mkString("Array(", ", ", ")"))
      .take(20)
      .foreach(println)

    classify_twitch_vertices(twRPGraph.vertices, tw_features)
    sc.getPersistentRDDs.foreach(tpl => tpl._2.unpersist(true))
    sc.stop()
  }



  def load_twitch_edges(sc: SparkContext): Graph[Float, Int] = {
    val path = "D:\\ChromeDownloads\\twitch_gamers\\large_twitch_edges.csv"

    val edges = sc.textFile(path, sc.defaultParallelism)
      .zipWithIndex()
      .filter(x => x._2 != 0)
      .keys
      .map(x => {
        val spl = x.split(",")
        Edge(spl(0).toLong, spl(1).toLong, 1)
      })
    Graph.fromEdges(edges, defaultValue = 1.0f)
  }

  def load_twitch_features(sc: SparkContext): RDD[TwitchFeatures] = {
    val path = "D:\\ChromeDownloads\\twitch_gamers\\large_twitch_features.csv"
    sc.textFile(path)
      .zipWithIndex()
      .filter(x => x._2 != 0)
      .keys
      .map(x => {
        val splits = x.split(",")
        TwitchFeatures(views = splits(0).toInt,
                      mature = splits(1).toInt,
                      numeric_id = splits(5).toLong,
                      language = splits(7),
                      affiliate = splits(8).toInt)
      })
  }

  case class TwitchFeatures(views: Int, mature: Int, numeric_id: Long, language: String, affiliate: Int){
    def englishOrNot: Int = if (language=="EN") 1 else 0
  }

  def classify_twitch_vertices(vertices: VertexRDD[Array[Float]], twitchFeatures: RDD[TwitchFeatures]) : Unit = {
    val idTarget: RDD[(Long, Int)] = twitchFeatures.keyBy(t => t.numeric_id).mapValues(t => t.affiliate)

    val data: RDD[(VertexId, LabeledPoint)] = vertices.join(idTarget)
      .mapValues(tpl => LabeledPoint(tpl._2.toFloat, Vectors.dense(tpl._1.map(_.toDouble))))

    classify(data)
    }
  def classify(data: RDD[(VertexId, LabeledPoint)]): Unit = {
    val std = new StandardScaler(withMean = true, withStd = true)

    val scaler = std.fit(data.values.map(lp => lp.features))
    val scaledData = data.mapValues(lp => LabeledPoint(lp.label, scaler.transform(lp.features)))
    val svm = new SVMWithSGD()

    svm.optimizer
      .setNumIterations(400)

    // Train a RandomForest model.
    // Empty categoricalFeaturesInfo indicates all features are continuous.
    val numClasses = 2
    val categoricalFeaturesInfo = Map[Int, Int]()
    val numTrees = 16
    val featureSubsetStrategy = "auto" // Let the algorithm choose.
    val impurity = "gini"
    val maxDepth = 30
    val maxBins = 32

    val dataCount = scaledData.count()
    val s = scaledData
      .keyBy(x => x._2.label)
      .countByKey()
      .map(dl => (dl._1, dl._2.toFloat / dataCount))


    val splits = scaledData
      .repartition(scaledData.getNumPartitions)
      .randomSplit(weights = Array(0.7, 0.3), seed = 42L)

    val train = splits(0).persist(StorageLevel.MEMORY_ONLY_SER)
    val test = splits(1).persist(StorageLevel.MEMORY_ONLY_SER)

    val model: SVMModel = svm.run(train.values)
//    val model = new LogisticRegressionWithLBFGS().setNumClasses(2).run(train.values)
//    val model = new NaiveBayes().run(train.values)

//    val model = RandomForest.trainClassifier(train.values, numClasses, categoricalFeaturesInfo,
//      numTrees, featureSubsetStrategy, impurity, maxDepth, maxBins)
    val predictionAndLabels = test.values.map(lp => (model.predict(lp.features), lp.label))

    val metrics = new BinaryClassificationMetrics(predictionAndLabels)

    // Precision by threshold
    val precision = metrics.precisionByThreshold()
    precision.collect().foreach { case (t, p) =>
      println(s"Threshold: $t, Precision: $p")
    }

    // Recall by threshold
    val recall = metrics.recallByThreshold()
    recall.collect().foreach { case (t, r) =>
      println(s"Threshold: $t, Recall: $r")
    }

    // Precision-Recall Curve
    val PRC = metrics.pr()


    // F-measure
    val f1Score = metrics.fMeasureByThreshold()
    f1Score.collect().foreach { case (t, f) =>
      println(s"Threshold: $t, F-score: $f, Beta = 1")
    }

    val beta = 0.5
    val fScore = metrics.fMeasureByThreshold(beta)
    fScore.collect().foreach { case (t, f) =>
      println(s"Threshold: $t, F-score: $f, Beta = 0.5")
    }

    // AUPRC
    val auPRC = metrics.areaUnderPR()
    println(s"Area under precision-recall curve = $auPRC")

    // Compute thresholds used in ROC and PR curves
    val thresholds = precision.map(_._1)

    // ROC Curve
    val roc = metrics.roc()

    // AUROC
    val auROC = metrics.areaUnderROC()
    println(s"Area under ROC = $auROC")


    // Instantiate metrics object
    val metrics2 = new MulticlassMetrics(predictionAndLabels)

    // Confusion matrix
    println("Confusion matrix:")
    println(metrics2.confusionMatrix)

    // Overall Statistics
    val accuracy = metrics2.accuracy
    println("Summary Statistics")
    println(s"Accuracy = $accuracy")

    // Precision by label
    val labels = metrics2.labels
    labels.foreach { l =>
      println(s"Precision($l) = " + metrics2.precision(l))
    }

    // Recall by label
    labels.foreach { l =>
      println(s"Recall($l) = " + metrics2.recall(l))
    }

    // False positive rate by label
    labels.foreach { l =>
      println(s"FPR($l) = " + metrics2.falsePositiveRate(l))
    }

    // F-measure by label
    labels.foreach { l =>
      println(s"F1-Score($l) = " + metrics2.fMeasure(l))
    }

    // Weighted stats
    println(s"Weighted precision: ${metrics2.weightedPrecision}")
    println(s"Weighted recall: ${metrics2.weightedRecall}")
    println(s"Weighted F1 score: ${metrics2.weightedFMeasure}")
    println(s"Weighted false positive rate: ${metrics2.weightedFalsePositiveRate}")

    predictionAndLabels.map(pl => {
      var pred = 0.0
      if ( pl._1 > 0.0) {
        pred = 1.0
      } else {
        pred = 0.0
      }
      ((pred, pl._2), 1.0)
    }).reduceByKey(_ + _)
      .collect()
      .foreach(println)
  }

}