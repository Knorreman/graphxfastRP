ThisBuild / version := "0.1.0-SNAPSHOT"

ThisBuild / scalaVersion := "2.13.10"
// https://mvnrepository.com/artifact/org.apache.spark/spark-graphx
libraryDependencies += "org.apache.spark" %% "spark-graphx" % "3.4.0"
// https://mvnrepository.com/artifact/org.apache.spark/spark-graphx
libraryDependencies += "org.apache.spark" %% "spark-mllib" % "3.4.0"
libraryDependencies += "org.scalanlp" %% "breeze" % "2.1.0"


lazy val root = (project in file("."))
  .settings(
    name := "graphxfastRP"
  )
