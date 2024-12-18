#!/bin/bash
# 允许脚本在失败时停止
set -e

# 使用宿主网络构建镜像
docker build --network=host -t ros2-jazzy-dev:latest -f Dockerfile .