## Run the container
1. Create a directory for model caching (only necessary once):  
   `mkdir -p ~/.torch/fvcore_cache`
2. Change to the *docker* directory of this repository:  
   `cd docker`
3. Create environment variable file to store the user and group ids (only necessary once):  
   `echo UID=$(id -u)>.env && echo GID=$(id -g)>>.env`
4. Run the docker container:  
   `docker-compose run detectron2`

## Rebuild the container 
Rebuilding the container is only necessary when *Dockerfile* has been changed. The initial build is done automatically.  
1. Change to the *docker* directory of this repository:  
   `cd docker`
2. Create environment variable file to store the user and group ids (only necessary once):  
   `echo UID=$(id -u)>.env && echo GID=$(id -g)>>.env`
3. Trigger the build:  
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
USER ${USER_NAME}
```

### Temporary
Use sudo (with your usual credentials) inside of the container. Changes will be lost, when the container is restarted.  

**Example:**  
`sudo apt-get update`  
`sudo apt-get install nano vim emacs`
`sudo pip install pandas`

