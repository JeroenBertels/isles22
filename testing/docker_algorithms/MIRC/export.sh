#!/usr/bin/env bash

./build.sh

docker save mirc | gzip -c > MIRC.tar.gz
