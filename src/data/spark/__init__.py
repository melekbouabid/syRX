"""Big Data jobs (PySpark) of Tujya: the C3PI dataset v2 and the larger RAG corpus.

Jobs (one DVC stage each, dvc.yaml; parameters in params.yaml under `bigdata`):

- `index`    ALLXML + directory_of_images -> one row per C3PI image, class and type
             statistics, class selection (at least N consumer photos), download list
- `quality`  downloaded images -> size, resolution, corrupt, brightness, perceptual hash,
             duplicates; rejected images listed with their reason
- `split`    train / val / test without leakage, stratified by class -> config/splits/c3pi_v2
- `augment`  offline augmentation of the rare classes, train split only
- `epillid`  overlap of the ePillID benchmark with C3PI and with the classes
- `rag`      RAG corpus: cleaning, deduplication (exact and MinHash) and chunking

They run on the Spark cluster of docker-compose.yml (`python -m src.data.spark.submit <job>`)
or on any machine with Java 17+ and pyspark (`python -m src.data.spark.run <job>`).
Docs: docs/module1/big_data.md.
"""
