# Spark for the Big Data jobs (src/data/spark, docs/module1/big_data.md): the official
# apache/spark image (Spark 4.1.1, Java, Python 3 and PySpark 4.1.1) plus the few Python
# packages the jobs import. No Java or winutils on the Windows host: the jobs run here.
# docker-compose.yml starts it as a standalone cluster (one master, one worker, profile
# `spark`) with the repository mounted on /opt/tujia, so the package `src` is importable by
# the driver and by the executors (PYTHONPATH).
FROM apache/spark:4.1.1-python3

USER root
COPY requirements.txt /tmp/requirements.txt
RUN pip3 install --no-cache-dir -r /tmp/requirements.txt && rm /tmp/requirements.txt

USER spark
# The repository first (package `src`), then PySpark and Py4J of the image, so plain `python3`
# (pytest, the health check) imports pyspark as spark-submit does.
ENV PYTHONPATH=/opt/tujia:/opt/spark/python:/opt/spark/python/lib/py4j-0.10.9.9-src.zip \
    PYSPARK_PYTHON=python3 \
    SPARK_NO_DAEMONIZE=true
WORKDIR /opt/tujia

HEALTHCHECK --interval=10s --timeout=5s --retries=20 \
    CMD python3 -c "import pyspark, PIL, pandas, pyarrow, yaml" || exit 1
