#!/bin/bash

pushd /root > /dev/null

if [ -d "spark" ]; then
  echo "Spark seems to be installed. Exiting."
  return
fi

# Github tag:
if [[ "$SPARK_VERSION" == *\|* ]]
then
  mkdir spark
  pushd spark > /dev/null
  git init
  repo=`python -c "print '$SPARK_VERSION'.split('|')[0]"`
  git_hash=`python -c "print '$SPARK_VERSION'.split('|')[1]"`
  git remote add origin $repo
  git fetch origin
  git checkout $git_hash
  cd spark-gpu
  ./make-distribution.sh -Dscala-2.11 -Phadoop-2.4 -Pyarn
  popd > /dev/null

# Pre-packaged spark version:
else
  case "$SPARK_VERSION" in
    2.0.0)
      wget http://s3.amazonaws.com/spark-gpu-public/spark-gpu-latest-bin-hadoop2.4.tgz
      if [ $? != 0 ]; then
        echo "ERROR: Unknown Spark version"
        return -1
      fi
    esac

  echo "Unpacking Spark"
  tar xvf spark-*.tgz > /tmp/spark-ec2_spark.log
  rm spark-*.tgz
  mv spark-gpu spark
fi
popd > /dev/null
