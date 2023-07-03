package fastp

import dev.ludovic.netlib.blas.BLAS
import org.apache.spark.SparkContext
import org.apache.spark.broadcast.Broadcast
import org.apache.spark.graphx.{EdgeContext, EdgeDirection, EdgeTriplet, Graph, PartitionStrategy, TripletFields, VertexId, VertexRDD}

import scala.util.Random

object FastRP {
  private lazy val blas: BLAS = BLAS.getInstance()

  def fastRP[A](graph: Graph[A, Double], dimensions: Int, weights: Array[Double], sparsity: Int = 3, r0: Double = 0.0): Graph[Array[Double], Double] = {
    val coeff = math.sqrt(sparsity)

    val initializedGraph: Graph[Array[Double], Double] = graph.mapVertices((id, _) => {
      val random = new Random(id)
      val seeds = generateSparseSeedBLAS(dimensions, coeff, sparsity, random)
//      val seeds = generateDenseSeedBLAS(dimensions, random)

//      normalize(seeds)
      seeds
    })

    val sc: SparkContext = graph.vertices.sparkContext
    val newWeights: Array[Double] = r0 +: weights
    val weightsBc: Broadcast[Array[Double]] = sc.broadcast(newWeights)

    val firstMsg = FastRPMessage(0, Array.fill(dimensions)(0.0), 1)

    val sendMsg = { (triplet: EdgeTriplet[FastRPVertex, Double]) => {
      Iterator((triplet.dstId, FastRPMessage(triplet.srcAttr.iteration,
                multiplyVectorByScalar(triplet.srcAttr.embedding, triplet.attr), 1)))
      }
    }

    val mergeMsg = { (msg1: FastRPMessage, msg2: FastRPMessage) => {
        FastRPMessage(msg1.iteration, addVectors(msg1.vector, msg2.vector), msg1.degreeCount + msg2.degreeCount)
      }
    }

    val vprog = { (_: VertexId, vertexAttr: FastRPVertex, msg: FastRPMessage) => {
      val avgVector = if (msg.iteration == 0) {
          vertexAttr.embedding
        } else {
          multiplyVectorByScalar(msg.vector, 1.0 / msg.degreeCount)
        }
      val wi = weightsBc.value(msg.iteration)
      val aggrEmbedding = multiplyVectorByScalar(normalize(avgVector), wi)

      FastRPVertex(vertexAttr.iteration + 1, addVectors(aggrEmbedding, vertexAttr.aggrEmbedding), avgVector)
      }
    }

    val pregelGraph: Graph[FastRPVertex, Double] = initializedGraph
      .mapVertices((_, v) => FastRPVertex(0, Array.fill(dimensions)(0.0), v))

    val finishedGraph: Graph[FastRPVertex, Double] = pregelGraph
      .pregel(initialMsg = firstMsg,
        maxIterations = weightsBc.value.length - 1,
        activeDirection = EdgeDirection.Either)(vprog, sendMsg, mergeMsg)

    finishedGraph
      .mapVertices((_, v) => v.aggrEmbedding)
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
        .take(k + 1)
        .foreach(println)

    }

    def euclideanDistance(v1: Array[Double], v2: Array[Double]): Double = {
      require(v1.length == v2.length, "Both input vectors should have the same length")

      val dotProduct = blas.ddot(v1.length, v1, 1, v2, 1)
      val norm1 = blas.dnrm2(v1.length, v1, 1)
      val norm2 = blas.dnrm2(v2.length, v2, 1)

      val euclideanDist: Double = Math.sqrt(norm1 * norm1 + norm2 * norm2 - 2 * dotProduct)
      euclideanDist
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

    def normalize(v: Array[Double]): Array[Double] = {
      val norm = blas.dnrm2(v.length, v, 1)
      val scale = if (norm == 0) 1.0 else 1.0 / norm
      val vcopy = v.clone()
      blas.dscal(v.length, scale, vcopy, 1)
      vcopy
    }

    def addVectors(a: Array[Double], b: Array[Double]): Array[Double] = {
      //    a.zip(b).map { case (x, y) => x + y }
      val bcopy = b.clone()
      blas.daxpy(a.length, 1.0, a, 1, bcopy, 1)
      bcopy
    }

  def multiplyVectorByScalar(v: Array[Double], scalar: Double): Array[Double] = {
    val vcopy = v.clone()
    blas.dscal(v.length, scalar, vcopy, 1)
    vcopy
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

  def generateDenseSeedBLAS(dimensions: Int, random: Random): Array[Double] = {
    val vec = Array.fill(dimensions)(0.0)
    (0 until dimensions).foreach(i => vec(i) = random.nextGaussian())
    vec
  }

  case class FastRPMessage(var iteration: Int, var vector: Array[Double], var degreeCount: Int)

  case class FastRPVertex(var iteration: Int, var aggrEmbedding: Array[Double], var embedding: Array[Double])
  }
