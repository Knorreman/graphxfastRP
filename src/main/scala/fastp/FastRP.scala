package fastp

import breeze.numerics.sqrt
import org.apache.spark.graphx.{EdgeContext, Graph, VertexId, VertexRDD}
import org.apache.spark.ml.linalg.Vectors

import scala.collection.mutable.ArrayBuffer
import scala.util.Random

object FastRP {
  def fastRP[A](graph: Graph[A, ?], dimensions: Int, weights: Array[Double]): Graph[Array[Double], ?] = {
    val sparsity = 30
    val coeff = math.sqrt(sparsity / 2.0)

    def generateSparseSeed(dimensions: Int, coeff: Double, sparsity: Int, random: Random): Array[Double] = {
      (1 to dimensions).map { _ =>
        val r = random.nextDouble()
        if (r < 0.5 / sparsity) coeff
        else if (r < 1.0 / sparsity) -coeff
        else 0.0
      }.toArray
    }

    def normalize(v: Array[Double]): Array[Double] = {
      val norm = math.sqrt(v.map(x => x * x).sum)
      if (norm == 0) v else v.map(x => x / norm)
    }

    var updatedGraph = graph.mapVertices((id, _) => {
      val random = new Random()
//      val seeds = (1 to dimensions).map(_ => random.nextGaussian()).toArray
      val seeds = generateSparseSeed(dimensions, coeff, sparsity, random)
      normalize(seeds)
    }).mapEdges(_ => 1.0)

    def addVectors(a: Array[Double], b: Array[Double]): Array[Double] = {
      a.zip(b).map { case (x, y) => x + y }
    }

    def multiplyVectorByScalar(v: Array[Double], scalar: Double): Array[Double] = {
      v.map(x => x * scalar)
    }

    val e = ArrayBuffer[(Graph[Array[Double], ?], Double)]()
//    val tmp_tpl = (updatedGraph, 0.1)
//    e += tmp_tpl
    for (i <- weights.indices) {
      // Update random projection vectors based on neighbors
      val msgs: VertexRDD[(Array[Double], Int)] = updatedGraph.aggregateMessages[(Array[Double], Int)](
        triplet => {
          // Send message to destination vertex with src vertex embedding scaled by edge attribute
          triplet.sendToDst((triplet.srcAttr.map(_ * triplet.attr), 1))
        },
        // Merge message
        (a, b) => (a._1.zip(b._1).map(x => x._1 + x._2), a._2 + b._2)
      )

      // Update vertex embeddings
      updatedGraph = updatedGraph.outerJoinVertices(msgs) {
        case (_, _, Some(msg)) => msg._1.map((x => x / msg._2))
        case (_, embedding, None) => embedding
      }

      val tmp_tpl = (updatedGraph, weights(i))
      e += tmp_tpl
    }
//
//    e(0)._1.vertices
//      .filter(v => v._1==352666384)
//      .mapValues(x => x.mkString("Array(", ", ", ")"))
//      .take(1).foreach(println)
//
//    e(1)._1.vertices
//      .filter(v => v._1 == 352666384)
//      .mapValues(x => x.mkString("Array(", ", ", ")"))
//      .take(1).foreach(println)
//
//    e(2)._1.vertices
//      .filter(v => v._1 == 352666384)
//      .mapValues(x => x.mkString("Array(", ", ", ")"))
//      .take(1).foreach(println)

    val result = e.map(tpl => tpl._1.mapVertices((_, arr) => multiplyVectorByScalar(normalize(arr), tpl._2)))
      .reduce((g1, g2) => {
        (g1.joinVertices(g2.vertices)((_, a1, a2) => {
          addVectors(a1, a2)
        }))
      })

    //      .mapValues(a => a._1.zip(a._2).map(x => x._1 + x._2)))
    //    updatedGraph.mapVertices((_, x) => normalize(x._1))
//    updatedGraph
    result
  }

  def query_knn(vectorRDD: VertexRDD[Array[Double]], domains: VertexRDD[String], k: Int = 10, query_array: Array[Double]): Unit = {
      val queryBc = vectorRDD.sparkContext.broadcast(query_array)

    vectorRDD.mapValues(arr => {
      val dist = cosineDistance(arr, queryBc.value)
      dist
    })
      .join(domains)
      .map(tpl => (tpl._2._1, (tpl._1, tpl._2._2)))
      .sortByKey(ascending = true)
      .take(k+1)
      .foreach(println)

  }
  def euclideanDistance(v1: Array[Double], v2: Array[Double]): Double = {
    require(v1.length == v2.length, "Both input vectors should have the same length")
    sqrt((v1 zip v2).map { case (a, b) => math.pow(a - b, 2) }.sum)
  }
  def dotProduct(v1: Array[Double], v2: Array[Double]): Double = {
    (v1 zip v2).map { case (a, b) => a * b }.sum
  }

  def magnitude(v: Array[Double]): Double = {
    sqrt(v.map(x => x*x).sum)
  }

  def cosineDistance(v1: Array[Double], v2: Array[Double]): Double = {
    require(v1.length == v2.length, "Both input vectors should have the same length")
    1.0 - (dotProduct(v1, v2) / (magnitude(v1) * magnitude(v2)))
  }
}
