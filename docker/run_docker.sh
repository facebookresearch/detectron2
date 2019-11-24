#!/bin/bash

# Create folder for model caching.
mkdir -p ~/.torch/fvcore_cache

# Run container.
export UID=$(id -u)
export GID=$(id -g)
docker-compose run detectron2
