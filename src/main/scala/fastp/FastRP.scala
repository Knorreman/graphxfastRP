package fastp

import breeze.linalg.{DenseVector, SparseVector, Vector}
import breeze.numerics.sqrt
import breeze.stats.distributions.{Gaussian, RandBasis, Uniform}
import dev.ludovic.netlib.blas.BLAS
import org.apache.spark.graphx.{EdgeContext, EdgeDirection, EdgeTriplet, Graph, TripletFields, VertexId, VertexRDD}

import scala.collection.mutable.ArrayBuffer
import scala.util.Random

object FastRP {
  private lazy val blas: BLAS = BLAS.getInstance()

  def fastRP[A](graph: Graph[A, ?], dimensions: Int, weights: Array[Float], sparsity: Int = 3, r0: Float = 0.0f): Graph[Array[Float], ?] = {
    val coeff = math.sqrt(sparsity).toFloat

    var updatedGraph = graph.mapVertices((id, _) => {
      val random = new Random(id)
            val seeds = generateSparseSeedBLAS(dimensions, coeff, sparsity, random)
//      val seeds = generateDenseSeedBLAS(dimensions, random)

      normalize(seeds)
    }).mapEdges(_ => 1.0)

    val intermediateResults = ArrayBuffer[(Graph[Array[Float], ?], Float)]()
    if (r0 != 0.0) {
      // Add source vector weight
      val g_tmp: Graph[Array[Float], _] = graph.mapVertices((_, attr) => attr.asInstanceOf[Array[Float]])

      intermediateResults += Tuple2(g_tmp, r0)
    }

    // Begin iterations
    for (i <- weights.indices) {
      val msgs: VertexRDD[(Array[Float], Float)] = updatedGraph.aggregateMessages[(Array[Float], Float)](
        triplet => {
          triplet.sendToDst((triplet.srcAttr, 1.0f))
          //          triplet.sendToSrc((triplet.dstAttr, 1.0))
        },
        (a, b) => (addVectors(a._1, b._1), a._2 + b._2),
        tripletFields = TripletFields.Src
      )

      updatedGraph = updatedGraph.outerJoinVertices(msgs) {
        case (_, _, Some(msg)) => multiplyVectorByScalar(msg._1, 1.0f / msg._2)
        case (_, embedding, None) => embedding
      }

      val wi = weights(i)
      if (wi != 0.0) {
        intermediateResults += Tuple2(updatedGraph, wi)
      }
    }

      val result: Graph[Array[Float], _] = intermediateResults
        .map(tpl => tpl._1.mapVertices((_, arr) => multiplyVectorByScalar(normalize(arr), tpl._2)))
        .reduceLeft((g1, g2) => {
            (g1.joinVertices(g2.vertices)((_, a1, a2) => {
              addVectors(a1, a2)
            }))
          })
      result
    }

    def query_knn(vectorRDD: VertexRDD[Array[Float]], domains: VertexRDD[String], k: Int = 10, query_array: Array[Float]): Unit = {
      val queryBc = vectorRDD.sparkContext.broadcast(query_array)

      vectorRDD.mapValues(arr => {
        val dist = cosineDistance(arr, queryBc.value)
        dist
      })
        .join(domains)
        .map(tpl => (tpl._2._1, (tpl._1, tpl._2._2)))
        .sortByKey(ascending = true)
        .take(k + 1)
        .foreach(println)

    }

    def euclideanDistance(v1: Array[Float], v2: Array[Float]): Float = {
      require(v1.length == v2.length, "Both input vectors should have the same length")
      //    sqrt((v1 zip v2).map { case (a, b) => math.pow(a - b, 2) }.sum)

      val dotProduct = blas.sdot(v1.length, v1, 1, v2, 1)
      val norm1 = blas.snrm2(v1.length, v1, 1)
      val norm2 = blas.snrm2(v2.length, v2, 1)

      val euclideanDist: Float = Math.sqrt(norm1 * norm1 + norm2 * norm2 - 2 * dotProduct).toFloat
      euclideanDist
    }

    private def dotProduct(v1: Array[Float], v2: Array[Float]): Float = {
      (v1 zip v2).map { case (a, b) => a * b }.sum
    }

    private def magnitude(v: Array[Float]): Float = {
      sqrt(v.map(x => x * x).sum)
    }

    def cosineDistance(v1: Array[Float], v2: Array[Float]): Float = {
      require(v1.length == v2.length, "Both input vectors should have the same length")

      val dotProduct = blas.sdot(v1.length, v1, 1, v2, 1)
      val norm1 = blas.snrm2(v1.length, v1, 1)
      val norm2 = blas.snrm2(v2.length, v2, 1)
      val cosineSimilarity = dotProduct / (norm1 * norm2)
      val cosineDistance = 1 - cosineSimilarity
      cosineDistance
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

    def normalize(v: Array[Float]): Array[Float] = {
      val norm = blas.snrm2(v.length, v, 1)
      val scale = if (norm == 0) 1.0f else 1.0f / norm
      val vcopy = v.clone()
      blas.sscal(v.length, scale, vcopy, 1)
      vcopy
    }

    def normalize(v: Vector[Float]): Vector[Float] = {
      breeze.linalg.normalize(v)
    }

    def addVectors(a: Array[Float], b: Array[Float]): Array[Float] = {
      //    a.zip(b).map { case (x, y) => x + y }
      val bcopy = b.clone()
      blas.saxpy(a.length, 1.0f, a, 1, bcopy, 1)
      bcopy
    }

    def addVectors(a: Vector[Float], b: Vector[Float]): Vector[Float] = {
      a + b
    }

    def multiplyVectorByScalar(v: Array[Float], scalar: Float): Array[Float] = {
      val vcopy = v.clone()
      blas.sscal(v.length, scalar, vcopy, 1)
      vcopy
    }

    def multiplyVectorByScalar(v: Vector[Float], scalar: Float): Vector[Float] = {
      v * scalar
    }

    def generateSparseSeed(dimensions: Int, coeff: Float, sparsity: Int, random: Random): Array[Float] = {
      (1 to dimensions).map { _ =>
        val r = random.nextFloat()
        if (r < 0.5f / sparsity) coeff
        else if (r < 1.0f / sparsity) -coeff
        else 0.0f
      }.toArray
    }

    def generateSparseSeedBreeze(dimensions: Int, coeff: Float, sparsity: Int, random: Random): SparseVector[Float] = {
      val vec = breeze.linalg.SparseVector.zeros[Float](dimensions)
      (0 until dimensions).foreach { i =>
        val r = random.nextFloat()
        if (r < 0.5f / sparsity) vec(i) = coeff
        else if (r < 1.0f / sparsity) vec(i) = -coeff
      }
      vec
    }

    def generateSparseSeedBLAS(dimensions: Int, coeff: Float, sparsity: Int, random: Random): Array[Float] = {

      val vec = Array.fill(dimensions)(0.0f)
      (0 until dimensions).foreach { i =>
        val r = random.nextFloat()
        if (r < 0.5f / sparsity) vec(i) = coeff
        else if (r < 1.0f / sparsity) vec(i) = -coeff
      }
      vec
    }

    def generateDenseSeedBreeze(dimensions: Int, random: Random): DenseVector[Double] = {
      val vec = DenseVector.rand[Double](dimensions, Gaussian(0, 1)(rand = RandBasis.withSeed(random.nextInt())))
      vec
    }

    def generateDenseSeedBLAS(dimensions: Int, random: Random): Array[Float] = {
      val vec = Array.fill(dimensions)(0.0f)
      (0 until dimensions).foreach(i => vec(i) = random.nextFloat())
      vec
    }

  }