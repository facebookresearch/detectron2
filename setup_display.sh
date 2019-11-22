#!/bin/bash
sudo apt-get install x11-xserver-utils
xhost +local:`docker inspect --format='{{ .Config.Hostname }}' detectron2`