#!/usr/bin/bash
set -e

export RMW_IMPLEMENTATION=rmw_zenoh_cpp
export ZENOH_CONFIG_OVERRIDE='mode="client";connect/endpoints=["tcp/127.0.0.1:7447"];transport/shared_memory/enabled=false'
