
## Use the container (with docker ≥ 19.03)

```
cd docker/
# Build:
docker build --build-arg USER_ID=$UID -t detectron2:v0 .
# Run:
docker run --gpus all -it \
	--shm-size=8gb --env="DISPLAY" --volume="/tmp/.X11-unix:/tmp/.X11-unix:rw" \
	--name=detectron2 detectron2:v0

# Grant docker access to host X server to show images
xhost +local:`docker inspect --format='{{ .Config.Hostname }}' detectron2`
```

## Use the container (with docker < 19.03)

Install docker-compose and nvidia-docker2, then run:
```
cd docker && USER_ID=$UID docker-compose run detectron2
```

#### Using a persistent cache directory

You can prevent models from being re-downloaded on every run,
by storing them in a cache directory.

To do this, add `--volume=$HOME/.torch/fvcore_cache:/tmp:rw` in the run command.

## Install new dependencies
Add the following to `Dockerfile` to make persistent changes.
```
RUN sudo apt-get update && sudo apt-get install -y vim
```
Or run them in the container to make temporary changes.
