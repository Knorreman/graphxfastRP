package commoncrawl

import commoncrawl.CommonCrawlDatasets.{CommonCrawlEdges, CommonCrawlVertices, Ranks, save_path10k, save_path200k}
import fastp.FastRP.{fastRP, query_knn}
import org.apache.spark.graphx.{Graph, PartitionStrategy, VertexId, VertexRDD}
import org.apache.spark.mllib.classification.{LogisticRegressionWithLBFGS, SVMModel, SVMWithSGD}
import org.apache.spark.mllib.evaluation.{BinaryClassificationMetrics, MulticlassMetrics}
import org.apache.spark.mllib.feature.StandardScaler
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.rdd.RDD
import org.apache.spark.storage.StorageLevel
import org.apache.spark.{SparkConf, SparkContext}

object CommonCrawlFastRP {

  def main(args: Array[String]): Unit = {
    val conf = new SparkConf()
      .setAppName("FastRPCommonCrawl")
      .setMaster("local[*]")
      .set("spark.local.dir", "H:\\sparklocal\\")
      .set("spark.driver.memory", "32g")
      .set("spark.rdd.compress", "true")
      .set("spark.serializer", "org.apache.spark.serializer.KryoSerializer")
      .registerKryoClasses(Array(classOf[Ranks], classOf[CommonCrawlVertices], classOf[CommonCrawlEdges]))

    val sc = new SparkContext(conf)
    sc.setLogLevel("WARN")

    val graph10k: Graph[String, Double] = CommonCrawlDatasets.load_graph[String, Double](sc, save_path10k, numPartitions = 16)

    println("www10k vertices:")
    println(graph10k.vertices.count())
    println("www10k edges:")
    println(graph10k.edges.count())

    graph10k
      .vertices
      .take(10)
      .foreach(println)

    val weights = Array(0.0, 0.0, 1.0, 1.0)
    val fastrp10k = fastRP(graph10k, 32, weights)

    fastrp10k
      .vertices
      .mapValues(x => x.mkString("Array(", ", ", ")"))
      .take(10)
      .foreach(println)

    classify_website(fastrp10k.vertices, graph10k.vertices)

    println("query")
    val query_id = graph10k.vertices.map(_.swap)
      .lookup("com.nytimes").head
    val query_vector = fastrp10k.vertices.lookup(query_id).head
    query_knn(fastrp10k.vertices, graph10k.vertices, 15, query_vector)

    println("query2")
    val query_id2 = graph10k.vertices.map(_.swap)
      .lookup("com.arsenal").head
    val query_vector2 = fastrp10k.vertices.lookup(query_id2).head
    query_knn(fastrp10k.vertices, graph10k.vertices, 15, query_vector2)

    println("query3")
    val query_id3 = graph10k.vertices.map(_.swap)
      .lookup("com.delta").head
    val query_vector3 = fastrp10k.vertices.lookup(query_id3).head
    query_knn(fastrp10k.vertices, graph10k.vertices, 15, query_vector3)

    println("query4")
    val query_id4 = graph10k.vertices.map(_.swap)
      .lookup("com.latimes").head
    val query_vector4 = fastrp10k.vertices.lookup(query_id4).head
    query_knn(fastrp10k.vertices, graph10k.vertices, 15, query_vector4)

    println("query5")
    val query_id5 = graph10k.vertices.map(_.swap)
      .lookup("com.mckinsey").head
    val query_vector5 = fastrp10k.vertices.lookup(query_id5).head
    query_knn(fastrp10k.vertices, graph10k.vertices, 15, query_vector5)

    println("query6")
    query_website("com.wikihow", graph10k.vertices, fastrp10k.vertices)
    println("query7")
    query_website("com.hardrock", graph10k.vertices, fastrp10k.vertices)
    println("query8")
    query_website("com.renegadehealth", graph10k.vertices, fastrp10k.vertices)
    println("query9")
    query_website("com.google", graph10k.vertices, fastrp10k.vertices)
    println("query10")
    query_website("se.aftonbladet", graph10k.vertices, fastrp10k.vertices)
  }

  def query_website(site: String,
                    vertices: VertexRDD[String],
                    fastRPvertices: VertexRDD[Array[Double]]): Unit = {

    val query_id = vertices.map(_.swap)
      .lookup(site).head
    val query_vector = fastRPvertices.lookup(query_id).head
    query_knn(fastRPvertices, vertices, 15, query_vector)
  }

  def classify_website(embeddings: VertexRDD[Array[Double]], websites: VertexRDD[String]): Unit = {
    val idAndDomain = websites
      .mapValues(web => web.split("\\.")(0))
      .cache()
    val idAndDomainMap = idAndDomain.values
      .distinct()
      .zipWithIndex()
      .collectAsMap()

    val idAndDomainMapHc = websites.sparkContext.broadcast(idAndDomainMap)

    val idAndLabel = idAndDomain.mapValues(domain => idAndDomainMapHc.value(domain))

    val idAndLabeledPoint = embeddings.join(idAndLabel)
      .mapValues(tpl => LabeledPoint(tpl._2.toDouble, Vectors.dense(tpl._1)))
    println("Number of classes: " + idAndDomainMap.size)

    classify(idAndLabeledPoint, idAndDomainMap.size)

  }

  def classify(data: RDD[(VertexId, LabeledPoint)], numClasses: Int): Unit = {
    val std = new StandardScaler(withMean = true, withStd = true)

    val scaler = std.fit(data.values.map(lp => lp.features))
    val scaledData = data.mapValues(lp => LabeledPoint(lp.label, scaler.transform(lp.features)))
    val svm = new SVMWithSGD()

    svm.optimizer
      .setNumIterations(400)

    // Train a RandomForest model.
    // Empty categoricalFeaturesInfo indicates all features are continuous.
    val categoricalFeaturesInfo = Map[Int, Int]()
    val numTrees = 16
    val featureSubsetStrategy = "auto" // Let the algorithm choose.
    val impurity = "gini"
    val maxDepth = 16
    val maxBins = 32

    val splits = scaledData
      .repartition(scaledData.getNumPartitions)
      .randomSplit(weights = Array(0.7, 0.3), seed = 42L)

    val train = splits(0).values.persist(StorageLevel.MEMORY_ONLY_SER)
    val test = splits(1).values.persist(StorageLevel.MEMORY_ONLY_SER)

//    val model: SVMModel = svm.run(train.values)
    val model = new LogisticRegressionWithLBFGS()
          .setNumClasses(numClasses)
          .run(train)
    //    val model = new NaiveBayes().run(train.values)

//    val model = RandomForest.trainClassifier(train, numClasses, categoricalFeaturesInfo,
//                                numTrees, featureSubsetStrategy, impurity, maxDepth, maxBins)
    val predictionAndLabels = test.map(lp => (model.predict(lp.features), lp.label))

    // Instantiate metrics object
    val metrics2 = new MulticlassMetrics(predictionAndLabels)

    // Overall Statistics
    val accuracy = metrics2.accuracy
    println("Summary Statistics")
    println(s"Accuracy = $accuracy")

    // Weighted stats
    println(s"Weighted precision: ${metrics2.weightedPrecision}")
    println(s"Weighted recall: ${metrics2.weightedRecall}")
    println(s"Weighted F1 score: ${metrics2.weightedFMeasure}")
    println(s"Weighted false positive rate: ${metrics2.weightedFalsePositiveRate}")

  }

}
