package commoncrawl

import org.apache.spark.graphx._
import org.apache.spark.rdd.RDD
import org.apache.spark.storage.StorageLevel
import org.apache.spark.{SparkConf, SparkContext}

import scala.reflect.ClassTag

object CommonCrawlDatasets {
  val RANKS_PATH = "H:\\commoncrawl\\ranks.txt"
  val EDGES_PATH = "H:\\commoncrawl\\edges.txt"
  val VERTICES_PATH = "H:\\commoncrawl\\vertices.txt"
  val save_path10k = "D:\\commoncrawl\\www10k\\"
  val save_path200k = "D:\\commoncrawl\\www200k\\"
  val save_path500k = "D:\\commoncrawl\\www500k\\"
  val save_path1m = "D:\\commoncrawl\\www1m\\"
  val save_path5m = "D:\\commoncrawl\\www5m\\"
  def main(args: Array[String]): Unit = {

    val conf = new SparkConf()
      .setAppName("FastRPCommonCrawl")
      .setMaster("local[*]")
      .set("spark.local.dir", "D:\\sparklocal\\")
      .set("spark.driver.memory", "16g")
      .set("spark.rdd.compress", "true")
      .set("spark.serializer", "org.apache.spark.serializer.KryoSerializer")
      .registerKryoClasses(Array(classOf[Ranks], classOf[CommonCrawlVertices], classOf[CommonCrawlEdges]))

    val sc = new SparkContext(conf)
    sc.setLogLevel("WARN")

    println("Creating 10k subset")
    sc.setJobDescription("Creating 10k subset")
    val ccGraph10k = create_subset(sc, hc_limit = 10_000)

    println("Num Vertices 10k:")
    println(ccGraph10k.vertices.count())

    println("Num edges 10k:")
    println(ccGraph10k.edges.count())

    println("Saving WWW-10K graph")
    sc.setJobDescription("Saving WWW-10K graph")
    save_graph(ccGraph10k, save_path10k)

    sc.getPersistentRDDs.foreach{ case (_, rdd) => rdd.unpersist(true) }

    println("Creating 200k subset")
    sc.setJobDescription("Creating 200k subset")
    val ccGraph200k = create_subset(sc, hc_limit = 200_000)

    println("Num Vertices 200k:")
    println(ccGraph200k.vertices.count())

    println("Num edges 200k:")
    println(ccGraph200k.edges.count())

    println("Saving WWW-200K graph")
    sc.setJobDescription("Saving WWW-200K graph")
    save_graph(ccGraph200k, save_path200k)

    sc.getPersistentRDDs.foreach{ case (_, rdd) => rdd.unpersist(true) }

    println("Creating 500k subset")
    sc.setJobDescription("Creating 500k subset")
    val ccGraph500k = create_subset(sc, hc_limit = 500_000)

    println("Num Vertices 500k:")
    println(ccGraph500k.vertices.count())

    println("Num edges 500k:")
    println(ccGraph500k.edges.count())

    println("Saving WWW-500K graph")
    sc.setJobDescription("Saving WWW-500K graph")
    save_graph(ccGraph500k, save_path500k)

    sc.getPersistentRDDs.foreach { case (_, rdd) => rdd.unpersist(true) }

    println("Creating 1M subset")
    sc.setJobDescription("Creating 1m subset")
    val ccGraph1m = create_subset(sc, hc_limit = 1_000_000)

    println("Num Vertices 1m:")
    println(ccGraph1m.vertices.count())

    println("Num edges 1m:")
    println(ccGraph1m.edges.count())

    println("Saving WWW-1M graph")
    sc.setJobDescription("Saving WWW-1M graph")
    save_graph(ccGraph1m, save_path1m)

    sc.getPersistentRDDs.foreach { case (_, rdd) => rdd.unpersist(true) }


    println("Creating 5m subset")
    sc.setJobDescription("Creating 5m subset")
    val ccGraph5m = create_subset(sc, hc_limit = 5_000_000)

    println("Num Vertices 5m:")
    println(ccGraph5m.vertices.count())

    println("Num edges 5m:")
    println(ccGraph5m.edges.count())

    println("Saving WWW-5M graph")
    sc.setJobDescription("Saving WWW-5M graph")
    save_graph(ccGraph5m, save_path5m)

    sc.getPersistentRDDs.foreach { case (_, rdd) => rdd.unpersist(true) }

    sc.stop()
  }

  def create_subset(sc: SparkContext, hc_limit: Long): Graph[String, Double] = {

    val ranks = load_ranks(sc, hc_limit)
    val vertices = load_vertices(sc)
    val filtered_vertices = vertices
      .keyBy(ccv => ccv.host)
      .join(ranks.keyBy(r => r.host_rev))

    val filtered_processed_vertices: RDD[(VertexId, String)] = filtered_vertices
      .values
      .map(tpl => (tpl._1.id, tpl._1.host))

    val edges: RDD[CommonCrawlEdges] = load_edges(sc)
    val graph2: Graph[String, Double] = Graph.fromEdges(
      edges.map(e => Edge(srcId = e.srcId, dstId = e.dstId, attr = 1.0)),
      defaultValue = "")

    val filtered_edges = edges
      .keyBy(e => e.srcId)
      .join(filtered_processed_vertices)
      .values
      .map(tpl => tpl._1)
      .keyBy(e => e.dstId)
      .join(filtered_processed_vertices)
      .values
      .map(tpl => tpl._1)

    val graph_edges = filtered_edges
      .map(e => Edge(srcId = e.srcId, dstId = e.dstId, attr = 1.0))
      .distinct()

    val ccGraph = Graph(filtered_processed_vertices, graph_edges,
      defaultVertexAttr = null.asInstanceOf[String],
      edgeStorageLevel = StorageLevel.MEMORY_AND_DISK,
      vertexStorageLevel = StorageLevel.MEMORY_AND_DISK)
      .filter(
        graph => {
          val degrees: VertexRDD[Int] = graph.degrees
          graph.outerJoinVertices(degrees) { (vid, data, deg) => deg.getOrElse(0) }
        },
        vpred = (vid: VertexId, deg: Int) => deg > 0
      )
      .partitionBy(PartitionStrategy.EdgePartition2D)

    ccGraph.vertices.take(10)
      .foreach(println)
    ccGraph
  }

  case class Ranks(hc_pos: Long, hc_val: Double, pr_pos: Long, pr_val: Double, host_rev: String)
  case class CommonCrawlVertices(id: Long, host: String)
  case class CommonCrawlEdges(srcId: Long, dstId: Long)

  def load_ranks(sc: SparkContext, hc_rank_limit: Long = 10000): RDD[Ranks] = {
    sc.textFile(RANKS_PATH)
      .mapPartitionsWithIndex { (idx, iter) => if (idx == 0) iter.drop(1) else iter }
      .map(s => {
        val splits = s.split("\t")
        Ranks(hc_pos = splits(0).toLong,
          hc_val = splits(1).toDouble,
          pr_pos = splits(2).toLong,
          pr_val = splits(3).toDouble,
          host_rev = splits(4))
      })
      .filter(sl => sl.hc_pos <= hc_rank_limit )
  }

  def load_edges(sc: SparkContext): RDD[CommonCrawlEdges] = {
    sc.textFile(EDGES_PATH)
      .map(s => {
        val splits = s.split("\t")
        CommonCrawlEdges(srcId = splits(0).toLong, dstId = splits(1).toLong)
      })
  }

  def load_vertices(sc: SparkContext): RDD[CommonCrawlVertices] = {
    sc.textFile(VERTICES_PATH)
      .map(s => {
        val splits = s.split("\t")
        CommonCrawlVertices(id = splits(0).toLong, host = splits(1))
      })
  }

  def save_graph[VD, ED](graph: Graph[VD, ED], path: String): Unit = {
    graph.vertices.saveAsObjectFile(path + "\\vertices")
    graph.edges.saveAsObjectFile(path + "\\edges")
  }

  def load_graph[VD: ClassTag, ED: ClassTag](sc: SparkContext, path: String, numPartitions: Int = 128, undirected: Boolean = true): Graph[VD, ED] = {
    val vertices = sc.objectFile[(VertexId, VD)](path + "\\vertices\\")
    val edgesBase = sc.objectFile[Edge[ED]](path + "\\edges\\")
    val edges = if (undirected) {
      val cached = edgesBase.cache()
      cached.union(cached.map(e => Edge(e.dstId, e.srcId, e.attr)))
    } else edgesBase

    Graph(vertices.repartition(numPartitions),
        edges.repartition(numPartitions),
        defaultVertexAttr = null.asInstanceOf[VD],
        edgeStorageLevel = StorageLevel.MEMORY_ONLY,
        vertexStorageLevel = StorageLevel.MEMORY_ONLY)
      .partitionBy(PartitionStrategy.EdgePartition1D, numPartitions)
  }

}
