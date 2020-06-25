#!/bin/bash

set -e

docker build . -t competitions-v1-compute-worker

mkdir -p /tmp/codalab

docker run \
       -v /var/run/docker.sock:/var/run/docker.sock \
       -v /tmp/codalab:/tmp/codalab \
       -d \
       --name compute_worker \
       --env BROKER_URL='pyamqp://77206c79-544b-4d71-b673-074a6b71497a:464289e9-9309-49cd-b7b9-18e0993f9ef5@competitions.codalab.org:5671/496f41ff-ce4e-4251-9197-bfcf36606e76' \
       --env BROKER_USE_SSL=True \
       --restart unless-stopped \
       --log-opt max-size=50m \
       --log-opt max-file=3 \
       competitions-v1-compute-worker
