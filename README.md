# graphxfastRP
## [fastRP](https://arxiv.org/pdf/1908.11512.pdf) algorithm implemented in graphx

# Build
    sbt clean package
# [Commoncrawl](https://commoncrawl.org/2017/05/hostgraph-2017-feb-mar-apr-crawls/) results
Download ranks.txt, vertices.txt and edges.txt. Point the *path_xx* to the correct files in CommonCrawlDatasets object.
Also change the *save_pathxxx* to where you want the final files. Then run the CommonCrawlDatasets object to preprocess the graph.

The paper uses top 10k and top 200k vertices based on Harmonic Centrality.
