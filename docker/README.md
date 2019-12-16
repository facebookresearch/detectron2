## Run the container
Change to the *docker* directory of this repository:
```
cd docker
USER_ID=$UID docker-compose run detectron2
```

#### Using a persistent cache directory
Prevents models to be re-downloaded on every run, by storing them in a cache directory.

`docker-compose run --volume=/path/to/cache:/tmp:rw detectron2`

## Rebuild the container
Rebuild the container  by `USER_ID=$UID docker-compose build detectron2`.
This is only necessary when `Dockerfile` has been changed. The initial build is done automatically.

## Install new dependencies
Add the following to `Dockerfile` to make persistent changes.
```
RUN sudo apt-get update && sudo apt-get install -y \
  nano vim emacs
RUN pip install --user pandas
```
Or run them in the container to make temporary changes.
