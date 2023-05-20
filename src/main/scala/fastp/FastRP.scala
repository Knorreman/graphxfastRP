package fastp

import breeze.linalg.{DenseVector, SparseVector, Vector}
import breeze.numerics.sqrt
import breeze.stats.distributions.{Gaussian, RandBasis}
import org.apache.spark.graphx.{EdgeContext, Graph, TripletFields, VertexId, VertexRDD}

import scala.collection.mutable.ArrayBuffer
import scala.util.Random

object FastRP {
  def fastRP[A](graph: Graph[A, ?], dimensions: Int, weights: Array[Double], sparsity: Int = 3, r0: Double = 0.0): Graph[Array[Double], ?] = {
    val coeff = math.sqrt(sparsity)

    var updatedGraph = graph.mapVertices((id, _) => {
      val random = new Random(id)
//      val seeds = (1 to dimensions).map(_ => random.nextGaussian()).toArray
      val seeds = generateSparseSeedBreeze(dimensions, coeff, sparsity, random)
//      val seeds = generateDenseSeedBreeze(dimensions, coeff, sparsity, random)

      normalize(seeds)
    }).mapEdges(_ => 1.0)

    val e = ArrayBuffer[(Graph[Vector[Double], ?], Double)]()
    // Add source vector weight
    if (r0 != 0.0) {
      val g_tmp = graph.mapVertices((_, attr ) => attr.asInstanceOf[Vector[Double]])

      val tmp_tpl = (g_tmp, r0)
      e += tmp_tpl
    }
    // Begin iterations
    for (i <- weights.indices) {
      val msgs: VertexRDD[(Vector[Double], Double)] = updatedGraph.aggregateMessages[(Vector[Double], Double)](
        triplet => {
          triplet.sendToDst((triplet.srcAttr * triplet.attr, 1.0))
        },
        (a, b) => (addVectors(a._1, b._1), a._2 + b._2),
        tripletFields = TripletFields.Src
      )

      updatedGraph = updatedGraph.outerJoinVertices(msgs) {
        case (_, _, Some(msg)) => msg._1 / msg._2
        case (_, embedding, None) => embedding
      }
      val wi = weights(i)
      if (wi != 0.0) {
        val tmp_tpl = (updatedGraph, weights(i))
        e += tmp_tpl
      }
    }

    val result = e.map(tpl => tpl._1.mapVertices((_, arr) => multiplyVectorByScalar(normalize(arr), tpl._2)))
      .reduce((g1, g2) => {
        (g1.joinVertices(g2.vertices)((_, a1, a2) => {
          addVectors(a1, a2)
        }))
      })
    result.mapVertices((_, arr) => arr.toArray)
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
  private def dotProduct(v1: Array[Double], v2: Array[Double]): Double = {
    (v1 zip v2).map { case (a, b) => a * b }.sum
  }

  private def magnitude(v: Array[Double]): Double = {
    sqrt(v.map(x => x*x).sum)
  }

  def cosineDistance(v1: Array[Double], v2: Array[Double]): Double = {
    require(v1.length == v2.length, "Both input vectors should have the same length")
    1.0 - (dotProduct(v1, v2) / (magnitude(v1) * magnitude(v2)))
  }

  def normalize(v: Array[Double]): Array[Double] = {
    val norm = math.sqrt(v.map(x => x * x).sum)
    if (norm == 0) v else v.map(x => x / norm)
  }

  def normalize(v: Vector[Double]): Vector[Double] = {
    breeze.linalg.normalize(v)
  }

  def addVectors(a: Array[Double], b: Array[Double]): Array[Double] = {
    a.zip(b).map { case (x, y) => x + y }
  }

  def addVectors(a: Vector[Double], b: Vector[Double]): Vector[Double] = {
    a + b
  }

  def multiplyVectorByScalar(v: Array[Double], scalar: Double): Array[Double] = {
    v.map(x => x * scalar)
  }

  def multiplyVectorByScalar(v: Vector[Double], scalar: Double): Vector[Double] = {
    v * scalar
  }

  def generateSparseSeed(dimensions: Int, coeff: Double, sparsity: Int, random: Random): Array[Double] = {
    (1 to dimensions).map { _ =>
      val r = random.nextDouble()
      if (r < 0.5 / sparsity) coeff
      else if (r < 1.0 / sparsity) -coeff
      else 0.0
    }.toArray
  }

  def generateSparseSeedBreeze(dimensions: Int, coeff: Double, sparsity: Int, random: Random): SparseVector[Double] = {
    val vec = breeze.linalg.SparseVector.zeros[Double](dimensions)
    (0 until dimensions).foreach { i =>
      val r = random.nextDouble()
      if (r < 0.5 / sparsity) vec(i) = coeff
      else if (r < 1.0 / sparsity) vec(i) = -coeff
    }
    vec
  }


  def generateDenseSeedBreeze(dimensions: Int, coeff: Double, sparsity: Int, random: Random): DenseVector[Double] = {
    val vec = DenseVector.rand[Double](dimensions, Gaussian(0, 1)(rand = RandBasis.withSeed(random.nextInt())))
    vec
  }

}
