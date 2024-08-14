#!/bin/bash
docker run --platform linux/amd64 \
    --rm -it \
    -v $PWD:/data \
    -u $(id -u):$(id -g) openjournals/inara \
    -o pdf,crossref \
    paper/paper.md
