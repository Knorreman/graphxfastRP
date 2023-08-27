package CompanyKG

import fastp.FastRP.{FastRPAMMessage, FastRPAMVertex, FastRPMessage, FastRPVertex, euclideanDistance, fastRPAM, fastRPPregel}
import org.apache.spark.graphx.{Edge, Graph, PartitionStrategy}
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
      .setMaster("local[10]")
      .set("spark.local.dir", "H:\\sparklocal\\")
      .set("spark.driver.memory", "50g")
      .set("spark.serializer", "org.apache.spark.serializer.KryoSerializer")
      .set("spark.graphx.pregel.checkpointInterval", "1")
      .set("spark.cleaner.referenceTracking.cleanCheckpoints", "true")
      .registerKryoClasses(Array(classOf[Array[Double]], classOf[Array[Float]],
        classOf[FastRPVertex], classOf[FastRPMessage], classOf[FastRPAMMessage], classOf[FastRPAMVertex]))

    val sc = new SparkContext(conf)
    sc.setLogLevel("WARN")

    sc.setCheckpointDir(checkpoint_dir)

    val path = ada2_path
    val graph = load_kg_graph(sc, path)

    println("Edges: ")
    println(graph.edges.count())

    println("Vertices: ")
    println(graph.vertices.count())

    val d = graph.vertices.values.first().length
    println("Number of features: " + d)

    val weights = Array(1.0, 1.0, 1.0)
    val fastRPGraph: Graph[Array[Double], Double] = fastRPAM[Array[Double]](graph, dimensions = d, weights = weights, r0 = 1.0)

    fastRPGraph.checkpoint()

    fastRPGraph.vertices
      .sortByKey(ascending = true)
      .take(5)
      .foreach(x => println(x._1, x._2.mkString("Array(", ", ", ")")))

    fastRPGraph.vertices
      .mapValues(_.mkString(" "))
      .coalesce(1)
      .sortByKey(ascending = true)
      .values
      .saveAsTextFile(path.replace(".", "_fastRP."))

      println("Saving done!")
    sc.stop(0)
  }

  def load_kg_graph(sc: SparkContext, path: String): Graph[Array[Double], Double] = {

    val numPartitions: Int = sc.defaultParallelism * 10
    val vertex_features = sc.textFile(path, numPartitions)
      .zipWithIndex()
      .map(_.swap)

    val vertices: RDD[(Long, Array[Double])] = vertex_features
      .mapValues(arr_str =>
        arr_str.split(" ").map(_.toDouble)
      )

    val edges = sc.textFile(edges_path, numPartitions)

    val edges_parsed: RDD[Edge[Double]] = edges
      .map(_.split(" "))
      .map(_.map(_.toLong))
      .map(x => (x(0), x(1)))
      .map(e => Edge(e._1, e._2, 1.0))

    Graph(vertices, edges_parsed,
        defaultVertexAttr = null.asInstanceOf[Array[Double]],
        edgeStorageLevel = StorageLevel.MEMORY_AND_DISK_SER,
        vertexStorageLevel = StorageLevel.MEMORY_AND_DISK_SER)
      .partitionBy(PartitionStrategy.EdgePartition1D, numPartitions)
    }
}
