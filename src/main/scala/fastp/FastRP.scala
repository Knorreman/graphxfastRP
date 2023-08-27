package fastp

import dev.ludovic.netlib.blas.BLAS
import org.apache.spark.SparkContext
import org.apache.spark.broadcast.Broadcast
import org.apache.spark.graphx.{EdgeContext, EdgeDirection, EdgeTriplet, Graph, PartitionStrategy, TripletFields, VertexId, VertexRDD}
import org.apache.spark.rdd.RDD

import scala.collection.mutable.ArrayBuffer
import scala.util.Random

object FastRP {
  private lazy val blas: BLAS = BLAS.getInstance()

  def fastRPAM[A](graph: Graph[A, Double], dimensions: Int, weights: Array[Double], sparsity: Int = 3, r0: Double = 0.0): Graph[Array[Double], Double] = {

    val initializedGraph: Graph[Array[Double], Double] = createInitialVector(graph, sparsity, dimensions)

    fastRPAMImpl(initializedGraph, weights, r0)
  }

  def fastRPPregel[A](graph: Graph[A, Double], dimensions: Int, weights: Array[Double], sparsity: Int = 3, r0: Double = 0.0): Graph[Array[Double], Double] = {

    val initializedGraph: Graph[Array[Double], Double] = createInitialVector(graph, sparsity, dimensions)

    fastRPPregelImpl(initializedGraph, dimensions, weights, r0)
  }

  private def createInitialVector[A](graph: Graph[A, Double], sparsity: Int, dimensions: Int): Graph[Array[Double], Double] = {
    val coeff = math.sqrt(sparsity)
    val initializedGraph: Graph[Array[Double], Double] = graph.mapVertices((id, v) => {

      val initArray: Array[Double] = v match {
        case arr: Array[Double] => arr
        case _ =>
          val random = new Random(id)
          val seeds = generateSparseSeedBLAS(dimensions, coeff, sparsity, random)
          seeds
      }
      initArray
    })
    initializedGraph
  }

  private def fastRPPregelImpl[A](initializedGraph: Graph[Array[Double], Double], dimensions: Int, weights: Array[Double], r0: Double = 0.0): Graph[Array[Double], Double] = {

    val sc: SparkContext = initializedGraph.vertices.sparkContext
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
      .mapVertices((_, v) => {
        v.aggrEmbedding
      })
  }

  private def fastRPAMImpl[A](initializedGraph: Graph[Array[Double], Double], weights: Array[Double], r0: Double = 0.0): Graph[Array[Double], Double] = {

    val partialResults: ArrayBuffer[(Graph[Array[Double], Double], Double)] = ArrayBuffer()
    var iterGraph: Graph[Array[Double], Double] = initializedGraph
    if (r0 != 0.0) {
      partialResults += Tuple2(iterGraph, r0)
    }

    for (weight <- weights) {
      val messages: VertexRDD[FastRPAMMessage] = iterGraph
        .mapVertices((_, v) => FastRPAMVertex(v))
        .aggregateMessages[FastRPAMMessage](sendMsg,
          mergeMsg,
          TripletFields.Src)

      val _vector : VertexRDD[Array[Double]] = messages
        .mapValues(v => multiplyVectorByScalar(v.vector, 1.0 / v.degreeCount))

      iterGraph = iterGraph.joinVertices(_vector)((_, _, vector) => vector)
      iterGraph.checkpoint()
      partialResults += Tuple2(iterGraph, weight)

    }

    partialResults
      .map(tpl => tpl._1.mapVertices((_, v) => multiplyVectorByScalar(normalize(v), tpl._2)))
      .reduceLeft((a, b) => a.joinVertices(b.vertices)((_, v1, v2) => addVectors(v1, v2)))
  }

  def sendMsg(edge: EdgeContext[FastRPAMVertex, Double, FastRPAMMessage]): Unit = {
    val vectorScaled = multiplyVectorByScalar(edge.srcAttr.vector, edge.attr)
    edge.sendToDst(FastRPAMMessage(vectorScaled, 1))
  }

  def mergeMsg(msg1: FastRPAMMessage, msg2: FastRPAMMessage): FastRPAMMessage = {
    FastRPAMMessage(addVectors(msg1.vector, msg2.vector), msg1.degreeCount + msg2.degreeCount)
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
    blas.dscal(v.length, scale, v, 1)
    v
  }

//  def normalize(v: Array[_]): Array[_] = {
//    v match {
//      case v: Array[Double] => {
//        val norm = math.sqrt(v.map(a => a * a).sum)
//        v.map(_ / norm)
//      }
//      case v: Array[Float] => {
//        val norm = math.sqrt(v.map(a => a * a).sum)
//        v.map(_ / norm)
//      }
//    }
//  }

//  def addVectors(a: Array[Double], b: Array[Double]): Array[Double] = {
//      val bcopy = b.clone()
//      blas.daxpy(a.length, 1.0, a, 1, bcopy, 1)
//      bcopy
//  }

  def addVectors(a: Array[Double], b: Array[Double]): Array[Double] = {
    a.zip(b).map { case (x, y) => (x + y) }
//    (a, b) match {
//      case (x: Array[Double], y: Array[Double]) => x.zip(y).map(w => w._1 + w._2)
//      case (x: Array[Float], y: Array[Float]) => x.zip(y).map(w => w._1 + w._2)
//      case (x: Array[Float], y: Array[Double]) => x.zip(y).map(w => w._1 + w._2).map(_.toDouble)
//      case (x: Array[Double], y: Array[Float]) => x.zip(y).map(w => w._1 + w._2).map(_.toDouble)
//      case _ => throw new IllegalArgumentException("Arrays must be both Double or both Float")
//    }
  }
//
  def multiplyVectorByScalar(v: Array[Double], scalar: Double): Array[Double] = {
    v.map(_ * scalar)
//    val vcopy = v.clone()
//    blas.dscal(v.length, scalar, vcopy, 1)
//    vcopy
  }

  // Trying to handle Doubles and Floats at the same time
//  def multiplyVectorByScalar(v: Array[_], scalar: _): Array[_] = {
//    (v, scalar) match {
//      case (v: Array[Double], scalar: Double) => v.map(_ * scalar)
//      case (v: Array[Float], scalar: Float) => v.map(_ * scalar)
//      case (v: Array[Float], scalar: Double) => v.map(_ * scalar)
//      case (v: Array[Double], scalar: Float) => v.map(_ * scalar)
//      case _ => throw new IllegalArgumentException("Arrays must be both Double or both Float")
//    }
//  }

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

  case class FastRPMessage(iteration: Int, vector: Array[Double], degreeCount: Int)

  case class FastRPVertex(iteration: Int, aggrEmbedding: Array[Double], embedding: Array[Double])

  case class FastRPAMVertex(vector: Array[Double])

  case class FastRPAMMessage(vector: Array[Double], degreeCount: Int)
  }
