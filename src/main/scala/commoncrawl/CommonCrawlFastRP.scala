package commoncrawl

import commoncrawl.CommonCrawlDatasets.{CommonCrawlEdges, CommonCrawlVertices, Ranks, save_path10k, save_path1m, save_path200k, save_path500k}
import dev.ludovic.netlib.blas.BLAS
import fastp.FastRP.{FastRPAMMessage, FastRPAMVertex, FastRPMessage, FastRPVertex, cosineDistance, fastRPAM, fastRPPregel, query_knn}
import org.apache.spark.broadcast.Broadcast
import org.apache.spark.graphx.{Graph, VertexId, VertexRDD}
import org.apache.spark.mllib.classification.SVMWithSGD
import org.apache.spark.mllib.evaluation.MulticlassMetrics
import org.apache.spark.mllib.feature.StandardScaler
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.mllib.stat.Statistics
import org.apache.spark.mllib.tree.RandomForest
import org.apache.spark.rdd.RDD
import org.apache.spark.storage.StorageLevel
import org.apache.spark.{SparkConf, SparkContext}

import scala.collection.mutable.ArrayBuffer
import scala.util.Random

object CommonCrawlFastRP {
  lazy val blas: BLAS = BLAS.getInstance()

  def main(args: Array[String]): Unit = {

    val conf = new SparkConf()
      .setAppName("FastRPCommonCrawl")
      .setMaster("local[*]")
      .set("spark.local.dir", "D:\\sparklocal\\")
      .set("spark.driver.memory", "45g")
      .set("spark.serializer", "org.apache.spark.serializer.KryoSerializer")
      .registerKryoClasses(Array(classOf[Ranks], classOf[CommonCrawlVertices], classOf[CommonCrawlEdges], classOf[Array[Double]],
        classOf[FastRPVertex], classOf[FastRPMessage], classOf[FastRPAMMessage], classOf[FastRPAMVertex]))

    val sc = new SparkContext(conf)
    sc.setLogLevel("WARN")

    val graph10k: Graph[String, Double] = CommonCrawlDatasets.load_graph[String, Double](sc, save_path10k, numPartitions = 16)

    println("www10k vertices:")
    println(graph10k.vertices.count())
    println("www10k edges:")
    println(graph10k.edges.count())

    val weights: Array[Double] = Array(0.0, 0.0, 1.0, 1.0, 0.5, 0.1, 0.025)
    val fastRP10k: Graph[Array[Double], Double] = fastRPAM(graph10k, 256, weights, r0=0.0001)

    val stats = Statistics.colStats(fastRP10k.vertices.map(x => Vectors.dense(x._2)))
    println(stats.max.toString)
    println(stats.min.toString)
    println(stats.variance.toString)
    println(stats.mean.toString)
    println(stats.normL1.toString)
    println(stats.normL2.toString)
    val statsBc = sc.broadcast(stats)

    val normedVertexVectors = fastRP10k
      .mapVertices { (_, x) =>
        val mean = statsBc.value.mean.toArray
        val variance = statsBc.value.variance.toArray
        val result = new Array[Double](x.length)
        var i = 0
        while (i < x.length) {
          result(i) = (x(i) - mean(i)) / Math.sqrt(variance(i))
          i += 1
        }
        result
      }
      .vertices

    normedVertexVectors
      .mapValues(x => x.mkString("Array(", ", ", ")"))
      .take(10)
      .foreach(println)

    classify_website(normedVertexVectors, graph10k.vertices)

    // Collect vertex name→id and id→vector maps once to avoid 26 separate RDD scan jobs
    val vertexNameToId: Map[String, VertexId] =
      graph10k.vertices.map { case (id, name) => (name, id) }.collectAsMap().toMap
    val vertexIdToVector: Map[VertexId, Array[Double]] =
      normedVertexVectors.collectAsMap().toMap

    def queryLocal(label: String, site: String): Unit = {
      println(label)
      val qVec = vertexIdToVector(vertexNameToId(site))
      query_knn(normedVertexVectors, graph10k.vertices, 10, qVec)
    }

    queryLocal("query", "com.nytimes")
    queryLocal("query2", "com.arsenal")
    queryLocal("query3", "com.delta")
    queryLocal("query4", "com.latimes")
    queryLocal("query5", "com.mckinsey")
    queryLocal("query6", "com.wikihow")
    queryLocal("query7", "com.hardrock")
    queryLocal("query8", "com.renegadehealth")
    queryLocal("query9", "com.google")
    queryLocal("query11", "org.python")
    queryLocal("query12", "com.twitter")
    queryLocal("query13", "org.4chan")
    queryLocal("query14", "com.apple")
    queryLocal("query10", "org.scala-lang")
    queryLocal("query16", "org.apache.spark")
    queryLocal("query17", "org.scikit-learn")
    queryLocal("query18", "org.tensorflow")

  }

  def query_website(site: String,
                    vertices: VertexRDD[String],
                    fastRPVertices: VertexRDD[Array[Double]]): Unit = {

    val query_id = vertices.map(_.swap)
      .lookup(site).head
    val query_vector = fastRPVertices.lookup(query_id).head
    query_knn(fastRPVertices, vertices, 10, query_vector)
  }

  def query_all_websites(vertices: VertexRDD[String],
                         fastRPVertices: VertexRDD[Array[Double]],
                         k: Int = 10): Unit = {
    val fastRPWithName = fastRPVertices
      .join(vertices, 16)
      .persist(StorageLevel.MEMORY_ONLY_SER)
    fastRPWithName.count()
    val cart = fastRPWithName
      .cartesian(fastRPWithName)
      .filter(tpl => tpl._1._1 != tpl._2._1)
      .keyBy(tpl => tpl._1._1)
      .mapValues(tpl => (tpl._1._2._2, tpl._2._2._2, cosineDistance(tpl._1._2._1, tpl._2._2._1)))
      .persist(StorageLevel.MEMORY_ONLY_SER)
    println("count of cartesian: " + cart.count())
    println("Starting first method")
    val now = System.nanoTime()
    cart
      .groupByKey()
      .flatMapValues(tpl3 => {
        tpl3.toArray.sortBy(_._3).zipWithIndex.take(k)
      })
      .values
      .take(10*k)
      .foreach(println)
    val now2 = System.nanoTime()
    println("Time taken: " + (now2 - now) / 1e9)

    println("Starting second method")
    val now3 = System.nanoTime()
    cart
      .aggregateByKey(ArrayBuffer.empty[(String, String, Double)])((arr, b) => {
          arr += b
        if (arr.length>k) {
          arr.sortBy(_._3).take(k)
        } else {
          arr
        }
      },
        (arr1, arr2) => {
          val conc = (arr1 ++ arr2)
          if (conc.length > k) {
            conc.sortBy(_._3).take(k)
          } else conc
        }
      )
      .flatMapValues(_.zipWithIndex)
      .values
      .take(10 * k)
      .foreach(println)

    val now4 = System.nanoTime()
    println("Time taken: " + (now4 - now3) / 1e9)

  }

  def classify_website(embeddings: VertexRDD[Array[Double]], websites: VertexRDD[String]): Unit = {
    val idAndDomain = websites
      .mapValues(web => web.split("\\.")(0))
//      .mapValues(s => if ( s == "com") 1 else 0)
      .cache()
    val idAndDomainMap = idAndDomain.values
      .distinct()
      .zipWithIndex()
      .collectAsMap()

//    idAndDomain
//      .values
//      .map(s => (s, 1))
//      .countByKey()
//      .foreach(println)

    val idAndDomainMapBc = websites.sparkContext.broadcast(idAndDomainMap)

    val idAndLabel = idAndDomain.mapValues(domain => idAndDomainMapBc.value(domain))

    val idAndLabeledPoint = embeddings.join(idAndLabel)
      .mapValues(tpl => LabeledPoint(tpl._2.toDouble, Vectors.dense(tpl._1)))
    println("Number of classes: " + idAndDomainMap.size)

    classify(idAndLabeledPoint, idAndDomainMap.size)

  }

  def classify(data: RDD[(VertexId, LabeledPoint)], numClasses: Int): Unit = {
    val std = new StandardScaler(withMean = true, withStd = true)

    val scaler = std.fit(data.values.map(lp => lp.features))
    val scaledData = data.mapValues(lp => LabeledPoint(lp.label, scaler.transform(lp.features))).cache()
    val svm = new SVMWithSGD()

    // Train a RandomForest model.
    // Empty categoricalFeaturesInfo indicates all features are continuous.
    val categoricalFeaturesInfo = Map[Int, Int]()
    val numTrees = 16
    val featureSubsetStrategy = "auto" // Let the algorithm choose.
    val impurity = "gini"
    val maxDepth = 8
    val maxBins = 32

    val splits = scaledData
      .randomSplit(weights = Array(0.7, 0.3), seed = 42L)

    val train = splits(0).values.persist(StorageLevel.MEMORY_ONLY_SER)
    val test = splits(1).values.persist(StorageLevel.MEMORY_ONLY_SER)

//    val model: SVMModel = svm.run(train)
//    val model = new LogisticRegressionWithLBFGS()
//          .setNumClasses(numClasses)
//          .run(train)
//        val model = new NaiveBayes().run(train)

    val model = RandomForest.trainClassifier(train, numClasses, categoricalFeaturesInfo,
                                numTrees, featureSubsetStrategy, impurity, maxDepth, maxBins)
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

  def random_project(vecRdd: RDD[(VertexId, Array[Float])], n_ht: Int = 10): RDD[(VertexId, Array[Double])] = {

    val arrLength = vecRdd.first()._2.length
    val rand = new Random(arrLength + 1)

    val matrix = Array.fill(arrLength * n_ht)(0.0)
    matrix.indices.foreach(i => {
      matrix(i) = rand.nextDouble()
    })

    val bVector = Array.fill(n_ht)(0.0)
    bVector.indices.foreach(i => {
      matrix(i) = rand.nextDouble()
    })
    val matrixBc: Broadcast[Array[Double]] = vecRdd.context.broadcast(matrix)
    val bVectorBc: Broadcast[Array[Double]] = vecRdd.context.broadcast(bVector)

    val r = 2.0
    vecRdd
      .mapValues(_.map(_.toDouble))
      .mapValues(v => matrix_vector_multiplication_add_vector(matrixBc.value, n_ht, arrLength, v, bVectorBc.value))
      .mapValues(v => v.map(_ / r).map(Math.floor))
  }

  private def matrix_vector_multiplication_add_vector(matrix: Array[Double], rows: Int, cols: Int,
                                                      vectorx: Array[Double],
                                                      vectory: Array[Double]): Array[Double] = {

    val vectorycopy = vectory.clone()

    blas.dgemv("n", rows, cols, 1.0, matrix, rows,
      vectorx, 1, 1.0, vectorycopy, 1)
    vectorycopy
  }

}
