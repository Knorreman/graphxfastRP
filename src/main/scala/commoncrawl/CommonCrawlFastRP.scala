package commoncrawl

import breeze.linalg.{*, DenseMatrix, DenseVector, InjectNumericOps, Matrix}
import breeze.numerics.{floor, round}
import breeze.optimize.DiffFunction.castOps
import breeze.stats.distributions.{Gaussian, RandBasis, Uniform}
import commoncrawl.CommonCrawlDatasets.{CommonCrawlEdges, CommonCrawlVertices, Ranks, save_path10k, save_path1m, save_path200k, save_path500k}
import dev.ludovic.netlib.blas.BLAS
import fastp.FastRP.{addVectors, cosineDistance, fastRP, normalize, query_knn}
import org.apache.spark.broadcast.Broadcast
import org.apache.spark.graphx.{Graph, VertexId, VertexRDD}
import org.apache.spark.mllib.classification.{LogisticRegressionWithLBFGS, SVMModel, SVMWithSGD}
import org.apache.spark.mllib.evaluation.MulticlassMetrics
import org.apache.spark.mllib.feature.StandardScaler
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.mllib.stat.Statistics
import org.apache.spark.mllib.tree.RandomForest
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.catalyst.expressions.ArrayFilter
import org.apache.spark.storage.StorageLevel
import org.apache.spark.{SparkConf, SparkContext}
//import org.apache.spark.ml.linalg.BLAS

import scala.collection.mutable.ArrayBuffer
import scala.util.Random

object CommonCrawlFastRP {
  lazy val blas: BLAS = BLAS.getInstance()

  def main(args: Array[String]): Unit = {

    val conf = new SparkConf()
      .setAppName("FastRPCommonCrawl")
      .setMaster("local[*]")
      .set("spark.local.dir", "D:\\sparklocal\\")
      .set("spark.driver.memory", "32g")
      .set("spark.rdd.compress", "true")
      .set("spark.serializer", "org.apache.spark.serializer.KryoSerializer")
      .registerKryoClasses(Array(classOf[Ranks], classOf[CommonCrawlVertices], classOf[CommonCrawlEdges], classOf[Array[Double]]))

    val sc = new SparkContext(conf)
    sc.setLogLevel("WARN")

    val graph10k: Graph[String, Double] = CommonCrawlDatasets.load_graph[String, Double](sc, save_path200k, numPartitions = 12)

    println("www10k vertices:")
    println(graph10k.vertices.count())
    println("www10k edges:")
    println(graph10k.edges.count())

    val weights = Array(0.0f, 0.0f, 1.0f, 0.0f, 1.0f)
    val fastRP10k = fastRP(graph10k, 128, weights)

    val stats = Statistics.colStats(fastRP10k.vertices.map(x => Vectors.dense(x._2.map(_.toDouble))))
    println(stats.max.toString)
    println(stats.min.toString)
    println(stats.variance.toString)
    println(stats.mean.toString)
    println(stats.normL1.toString)
    println(stats.normL2.toString)
    val statsBc = sc.broadcast(stats)

    val normedVertexVectors = fastRP10k
      .mapVertices((_, x) => addVectors(x, statsBc.value.mean.toArray.map(-_.toFloat)))
      .mapVertices((_, x) => x.zip(statsBc.value.variance.toArray).map(x => x._1/Math.sqrt(x._2).toFloat))

    normedVertexVectors
      .vertices
      .mapValues(x => x.mkString("Array(", ", ", ")"))
      .take(10)
      .foreach(println)

    val dCount = normedVertexVectors.vertices.map(_._2.toSeq).distinct().count()
    println("distinct arrays " + dCount)

    val lsh = random_project(normedVertexVectors.vertices, 16).mapValues(_.map(_.toFloat))

    val bucketCount = lsh
      .keyBy(_._2.toSeq)
      .mapValues(_ => 1)
      .reduceByKey(_ + _)
      .count()

    println("BucketCount " + bucketCount)

    lsh
      .keyBy(_._2.toSeq)
      .mapValues(_ => 1)
      .reduceByKey(_ + _)
      .sortBy(t => t._2, ascending = false)
      .take(20)
      .foreach(println)

//    val bkm = new BisectingKMeans()
//    bkm.setK(fastRP10k.vertices.count().toInt)
//    bkm.setDistanceMeasure(DistanceMeasure.EUCLIDEAN)
//    bkm.setMaxIterations(3)
//    bkm.setSeed(42L)
//
//    val bkmModel = bkm.run(fastRP10k.vertices.map(v => Vectors.dense(v._2)))
//    println("number of centers: " + bkmModel.clusterCenters.length)
//    fastRP10k.vertices
//      .join(graph10k.vertices)
//      .map(t => (t._2._2, t._2._1))
//      .map(v => (bkmModel.predict(Vectors.dense(v._2)), Array(v._1)))
//      .reduceByKey(_++_)
//      .sortBy(t => t._2.length, ascending = false)
//      .mapValues(x => x.mkString("Array(", ", ", ")"))
//      .take(20)
//      .foreach(println)

//    query_all_websites(graph10k.vertices, fastRP10k.vertices, 5)
//    System.exit(0)

    classify_website(normedVertexVectors.vertices.mapValues(_.map(_.toDouble)), graph10k.vertices)

    println("query")
    val query_id = graph10k.vertices.map(_.swap)
      .lookup("com.nytimes").head
    val query_vector = normedVertexVectors.vertices.lookup(query_id).head
    query_knn(normedVertexVectors.vertices, graph10k.vertices, 10, query_vector)

    println("query_lsh")
    val query_id_lsh = graph10k.vertices.map(_.swap)
      .lookup("com.nytimes").head
    val query_vector_lsh = lsh.mapValues(_.toArray).lookup(query_id_lsh).head
    query_knn(VertexRDD(lsh.mapValues(_.toArray)), graph10k.vertices, 10, query_vector_lsh)

    println("query2")
    val query_id2 = graph10k.vertices.map(_.swap)
      .lookup("com.arsenal").head
    val query_vector2 = normedVertexVectors.vertices.lookup(query_id2).head
    query_knn(normedVertexVectors.vertices, graph10k.vertices, 10, query_vector2)

    println("query_lsh2")
    val query_id_lsh2 = graph10k.vertices.map(_.swap)
      .lookup("com.arsenal").head
    val query_vector_lsh2 = lsh.mapValues(_.toArray).lookup(query_id_lsh2).head
    query_knn(VertexRDD(lsh.mapValues(_.toArray)), graph10k.vertices, 10, query_vector_lsh2)


    println("query3")
    val query_id3 = graph10k.vertices.map(_.swap)
      .lookup("com.delta").head
    val query_vector3 = fastRP10k.vertices.lookup(query_id3).head
    query_knn(fastRP10k.vertices, graph10k.vertices, 10, query_vector3)

    println("query_lsh3")
    val query_id_lsh3 = graph10k.vertices.map(_.swap)
      .lookup("com.delta").head
    val query_vector_lsh3 = lsh.mapValues(_.toArray).lookup(query_id_lsh3).head
    query_knn(VertexRDD(lsh.mapValues(_.toArray)), graph10k.vertices, 10, query_vector_lsh3)

    println("query4")
    val query_id4 = graph10k.vertices.map(_.swap)
      .lookup("com.latimes").head
    val query_vector4 = fastRP10k.vertices.lookup(query_id4).head
    query_knn(fastRP10k.vertices, graph10k.vertices, 10, query_vector4)

    println("query_lsh4")
    val query_id_lsh4 = graph10k.vertices.map(_.swap)
      .lookup("com.latimes").head
    val query_vector_lsh4 = lsh.mapValues(_.toArray).lookup(query_id_lsh4).head
    query_knn(VertexRDD(lsh.mapValues(_.toArray)), graph10k.vertices, 10, query_vector_lsh4)

    println("query5")
    val query_id5 = graph10k.vertices.map(_.swap)
      .lookup("com.mckinsey").head
    val query_vector5 = fastRP10k.vertices.lookup(query_id5).head
    query_knn(fastRP10k.vertices, graph10k.vertices, 10, query_vector5)

    println("query_lsh5")
    val query_id_lsh5 = graph10k.vertices.map(_.swap)
      .lookup("com.mckinsey").head
    val query_vector_lsh5 = lsh.mapValues(_.toArray).lookup(query_id_lsh5).head
    query_knn(VertexRDD(lsh.mapValues(_.toArray)), graph10k.vertices, 10, query_vector_lsh5)

    System.exit(0)
    println("query6")
    query_website("com.wikihow", graph10k.vertices, fastRP10k.vertices)
    println("query7")
    query_website("com.hardrock", graph10k.vertices, fastRP10k.vertices)
    println("query8")
    query_website("com.renegadehealth", graph10k.vertices, fastRP10k.vertices)
    println("query9")
    query_website("com.google", graph10k.vertices, fastRP10k.vertices)
    println("query10")
    query_website("se.aftonbladet", graph10k.vertices, fastRP10k.vertices)
    println("query11")
    query_website("org.python", graph10k.vertices, fastRP10k.vertices)
    println("query12")
    query_website("com.twitter", graph10k.vertices, fastRP10k.vertices)
    println("query13")
    query_website("org.4chan", graph10k.vertices, fastRP10k.vertices)
    println("query14")
    query_website("com.apple", graph10k.vertices, fastRP10k.vertices)
    println("query10")
    query_website("org.scala-lang", graph10k.vertices, fastRP10k.vertices)
    println("query16")
    query_website("org.apache.spark", graph10k.vertices, fastRP10k.vertices)
    println("query17")
    query_website("org.scikit-learn", graph10k.vertices, fastRP10k.vertices)
    println ("query18")
    query_website("org.tensorflow", graph10k.vertices, fastRP10k.vertices)
    println("query19")
    query_website("org.pytorch", graph10k.vertices, fastRP10k.vertices)


  }

  def query_website(site: String,
                    vertices: VertexRDD[String],
                    fastRPvertices: VertexRDD[Array[Float]]): Unit = {

    val query_id = vertices.map(_.swap)
      .lookup(site).head
    val query_vector = fastRPvertices.lookup(query_id).head
    query_knn(fastRPvertices, vertices, 10, query_vector)
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
    val scaledData = data.mapValues(lp => LabeledPoint(lp.label, scaler.transform(lp.features)))
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
      .repartition(scaledData.getNumPartitions)
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
