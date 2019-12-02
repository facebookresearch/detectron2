## Run the container
Change to the *docker* directory of this repository:  
`cd docker`
   
### as non-root user (recommended)
`docker-compose run detectron2`

### as root user
`docker-compose -f docker-compose.yml -f docker-compose.run_as_root.yml run detectron2`

### with a persistent cache directory
Prevents models to be re-downloaded on every run, by storing them in your */tmp* directory.

`docker-compose -f docker-compose.yml -f docker-compose.persistent_cache.yml run detectron2`

## Rebuild the container 
Rebuilding the container is only necessary when *Dockerfile* has been changed. The initial build is done automatically.  
1. Change to the *docker* directory of this repository:  
   `cd docker`
2. Trigger the build:  
   `docker-compose build detectron2`
   
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
Use sudo (with your usual credentials) inside of the container. Changes will be lost, when the container is restarted.  

**Example:**  
`sudo apt-get update`  
`sudo apt-get install nano vim emacs`  
`sudo pip install pandas`

