## Run the container
Change to the *docker* directory of this repository:
```
cd docker
docker-compose run detectron2
```

#### Using a persistent cache directory
Prevents models to be re-downloaded on every run, by storing them in a cache directory.

`docker-compose run --volume=/path/to/cache:/tmp:rw detectron2`

## Rebuild the container
Rebuild the container  by `docker-compose build detectron2`.
This is only necessary when `Dockerfile` has been changed. The initial build is done automatically.

## Install new dependencies
### Persistent
Add the dependencies at the end of *Dockerfile*.

**Example:**
```
...
# Customization
USER root
RUN apt-get update && apt-get install -y \
  nano vim emacs
RUN pip install pandas
USER appuser
```

### Temporary
Use sudo inside the container. Changes will be lost when the container is restarted.

**Example:**
`sudo apt-get update`
`sudo apt-get install nano vim emacs`
`sudo pip install pandas`
