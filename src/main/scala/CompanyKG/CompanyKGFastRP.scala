package CompanyKG

import fastp.FastRP.{FastRPAMMessage, FastRPAMVertex, FastRPMessage, FastRPVertex, euclideanDistance, fastRPAM, fastRPPregel}
import org.apache.spark.graphx.{Edge, Graph, GraphXUtils, PartitionStrategy}
import org.apache.spark.rdd.RDD
import org.apache.spark.storage.StorageLevel
import org.apache.spark.{SparkConf, SparkContext}

object CompanyKGFastRP {

  private val root_path = "H:\\CompanyKGData\\"
  private val path_prefix = root_path + "nodes_feature_"
  private val msbert_path = path_prefix + "msbert.txt"
  private val pause_path = path_prefix + "pause.txt"
  private val ada2_path = path_prefix + "ada2.txt"
  private val simcse_path = path_prefix + "simcse.txt"
  private val edges_path = root_path + "edges.txt"

  private val checkpoint_dir = root_path + "checkpoints_\\"

  def main(args: Array[String]): Unit = {

    val conf = new SparkConf()
      .setAppName("FastRPKG")
      .setMaster("local[13]")
      .set("spark.local.dir", "D:\\sparklocal\\")
      .set("spark.driver.memory", "50g")
      .set("spark.executor.memory", "50g")
      .set("spark.memory.fraction", "0.85")
      .set("spark.memory.storageFraction", "0.4")
      .set("spark.memory.offHeap.enabled", "true")
      .set("spark.memory.offHeap.size", "4g")
      .set("spark.serializer", "org.apache.spark.serializer.KryoSerializer")
      .set("spark.graphx.pregel.checkpointInterval", "1")
      .set("spark.cleaner.referenceTracking.cleanCheckpoints", "true")
      .registerKryoClasses(Array(classOf[Array[Double]], classOf[Array[Float]],
        classOf[Vector[Double]], classOf[Vector[Object]], classOf[String],
        classOf[FastRPVertex], classOf[FastRPMessage], classOf[FastRPAMMessage], classOf[FastRPAMVertex]))

    GraphXUtils.registerKryoClasses(conf)

    val sc = new SparkContext(conf)
    sc.setLogLevel("WARN")

    sc.setCheckpointDir(checkpoint_dir)

    val path = msbert_path
    val graph = load_kg_graph(sc, path)

    println("Edges: ")
    println(graph.edges.count())

    println("Vertices: ")
    println(graph.vertices.count())

    val initVectorOrNot: Boolean = true
//    val d = graph.vertices.values.first().length
    val d = if (!initVectorOrNot) 8 else graph.vertices.values.first().length
    println("Number of features: " + d)

    val weights = Array(0.75, 0.5, 0.25)
    val fastRPGraph: Graph[Array[Double], Double] = fastRPAM[Array[Double]](graph,
      dimensions = d,
      weights = weights,
      r0 = 1.0,
      initOrNot = initVectorOrNot)

    fastRPGraph.vertices
      .sortByKey(ascending = true)
      .take(10)
      .foreach(x => println(x._1, x._2.mkString("Array(", ", ", ")")))

    val extraString = if (initVectorOrNot) s"_fastRP_${weights.mkString("(", "_", ")")}."
      else "_" + d + s"_noInit_fastRP_${weights.mkString("(", "_", ")")}."

    fastRPGraph.vertices
      .mapValues(_.mkString(" "))
      .values
      .saveAsTextFile(path.replace(".", extraString))

      println("Saving done!")
    sc.stop(0)
  }

  def load_kg_graph(sc: SparkContext, path: String, undirected: Boolean = true): Graph[Array[Double], Double] = {

    val vertex_features = sc.textFile(path)
      .zipWithIndex()
      .map(_.swap)

    val vertices: RDD[(Long, Array[Double])] = vertex_features
      .mapValues(arr_str =>
        arr_str.split(" ").map(_.toDouble)
      )

    val numPartitions: Int = vertices.getNumPartitions
    val edges = sc.textFile(edges_path, numPartitions)

    val edges_base: RDD[Edge[Double]] = edges.map { line =>
      val parts = line.split(" ")
      Edge(parts(0).toLong, parts(1).toLong, 1.0)
    }

    // Make the graph undirected
    val edges_parsed: RDD[Edge[Double]] = if (undirected) {
      val cached = edges_base.cache()
      cached.union(cached.map(e => Edge(e.dstId, e.srcId, e.attr)))
    } else edges_base

    Graph(vertices, edges_parsed,
        defaultVertexAttr = null.asInstanceOf[Array[Double]],
        edgeStorageLevel = StorageLevel.MEMORY_AND_DISK_SER,
        vertexStorageLevel = StorageLevel.MEMORY_AND_DISK_SER)
      .partitionBy(PartitionStrategy.EdgePartition1D, numPartitions)
    }
}
