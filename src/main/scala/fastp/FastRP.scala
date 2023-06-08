package fastp

import breeze.linalg.{DenseVector, SparseVector, Vector}
import breeze.numerics.sqrt
import breeze.stats.distributions.{Gaussian, RandBasis, Uniform}
import dev.ludovic.netlib.blas.BLAS
import org.apache.spark.graphx.{EdgeContext, Graph, TripletFields, VertexId, VertexRDD}

import scala.collection.mutable.ArrayBuffer
import scala.util.Random

object FastRP {
  private lazy val blas: BLAS = BLAS.getInstance()
  def fastRP[A](graph: Graph[A, ?], dimensions: Int, weights: Array[Double], sparsity: Int = 3, r0: Double = 0.0): Graph[Array[Double], ?] = {
    val coeff = math.sqrt(sparsity)

    var updatedGraph = graph.mapVertices((id, _) => {
      val random = new Random()
//      val seeds = generateSparseSeedBLAS(dimensions, coeff, sparsity, random)
      val seeds = generateDenseSeedBLAS(dimensions, random)

      normalize(seeds)
    }).mapEdges(_ => 1.0)

    val intermediateResults = ArrayBuffer[(Graph[Array[Double], ?], Double)]()
    if (r0 != 0.0) {
      // Add source vector weight
      val g_tmp: Graph[Array[Double], _] = graph.mapVertices((_, attr) => attr.asInstanceOf[Array[Double]])

      intermediateResults += Tuple2(g_tmp, r0)
    }

    // Begin iterations
    for (i <- weights.indices) {
      val msgs: VertexRDD[(Array[Double], Double)] = updatedGraph.aggregateMessages[(Array[Double], Double)](
        triplet => {
          triplet.sendToDst((triplet.srcAttr, 1.0))
//          triplet.sendToSrc((triplet.dstAttr, 1.0))
        },
        (a, b) => (addVectors(a._1, b._1), a._2 + b._2),
        tripletFields = TripletFields.All
      )

      updatedGraph = updatedGraph.outerJoinVertices(msgs) {
        case (_, _, Some(msg)) => multiplyVectorByScalar(msg._1, 1.0/msg._2)
        case (_, embedding, None) => embedding
      }

      val wi = weights(i)
      if (wi != 0.0) {
        intermediateResults += Tuple2(updatedGraph, wi)
      }
    }

    val result: Graph[Array[Double], _] = intermediateResults
      .map(tpl => tpl._1.mapVertices((_, arr) => multiplyVectorByScalar(normalize(arr), tpl._2)))
      .reduce((g1, g2) => {
        (g1.joinVertices(g2.vertices)((_, a1, a2) => {
          addVectors(a1, a2)
        }))
      })
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
//    sqrt((v1 zip v2).map { case (a, b) => math.pow(a - b, 2) }.sum)

    val dotProduct = blas.ddot(v1.length, v1, 1, v2, 1)
    val norm1 = blas.dnrm2(v1.length, v1, 1)
    val norm2 = blas.dnrm2(v2.length, v2, 1)

    val euclideanDist = Math.sqrt(norm1 * norm1 + norm2 * norm2 - 2 * dotProduct)
    euclideanDist
  }
  private def dotProduct(v1: Array[Double], v2: Array[Double]): Double = {
    (v1 zip v2).map { case (a, b) => a * b }.sum
  }

  private def magnitude(v: Array[Double]): Double = {
    sqrt(v.map(x => x*x).sum)
  }

  def cosineDistance(v1: Array[Double], v2: Array[Double]): Double = {
    require(v1.length == v2.length, "Both input vectors should have the same length")

    val dotProduct = blas.ddot(v1.length, v1, 1, v2, 1)
    val norm1 = blas.dnrm2(v1.length, v1, 1)
    val norm2 = blas.dnrm2(v2.length, v2, 1)
    val cosineSimilarity = dotProduct / (norm1 * norm2)
    val cosineDistance = 1 - cosineSimilarity
    cosineDistance
  }

  def normalize(v: Array[Double]): Array[Double] = {
    val norm = blas.dnrm2(v.length, v, 1)
    val scale = if (norm == 0) 1.0 else 1.0 / norm
    val vcopy = v.clone()
    blas.dscal(v.length, scale, vcopy, 1)
    vcopy
  }

  def normalize(v: Vector[Double]): Vector[Double] = {
    breeze.linalg.normalize(v)
  }

  def addVectors(a: Array[Double], b: Array[Double]): Array[Double] = {
//    a.zip(b).map { case (x, y) => x + y }
    val bcopy = b.clone()
    blas.daxpy(a.length, 1.0, a, 1, bcopy, 1)
    bcopy
  }

  def addVectors(a: Vector[Double], b: Vector[Double]): Vector[Double] = {
    a + b
  }

  def multiplyVectorByScalar(v: Array[Double], scalar: Double): Array[Double] = {
    val vcopy = v.clone()
    blas.dscal(v.length, scalar, vcopy, 1)
    vcopy
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

  def generateSparseSeedBLAS(dimensions: Int, coeff: Double, sparsity: Int, random: Random): Array[Double] = {

    val vec = Array.fill(dimensions)(0.0)
    (0 until dimensions).foreach { i =>
      val r = random.nextDouble()
      if (r < 0.5 / sparsity) vec(i) = coeff
      else if (r < 1.0 / sparsity) vec(i) = -coeff
    }
    vec
  }

  def generateDenseSeedBreeze(dimensions: Int, random: Random): DenseVector[Double] = {
    val vec = DenseVector.rand[Double](dimensions, Gaussian(0, 1)(rand = RandBasis.withSeed(random.nextInt())))
    vec
  }

  def generateDenseSeedBLAS(dimensions: Int, random: Random): Array[Double] = {
    val vec = Array.fill(dimensions)(0.0)
    (0 until dimensions).foreach(i => vec(i) = random.nextDouble())
    vec
  }

}
