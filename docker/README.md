To run the detectron2 docker container:
1. Create a folder for model caching (only necessary on initial run):  
   `mkdir -p ~/.torch/fvcore_cache`
2. Create environment variables to store the user and group ids (always necessary):  
   `export UID=$(id -u)`  
   `export GID=$(id -g)`
3. Change into the docker directory:  
   `cd docker`
4. Run the docker container:  
   `docker-compose run detectron2`
   
To rebuild the docker container (only necessary when Dockerfile has been changed; initial build is done automatically):  
`docker-compose build detectron2`
