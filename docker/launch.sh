d2repo=/home/caishenghang/detectron2
extra=""
extra+=" -v /dataset:/dataset"
USER_ID=$UID docker-compose run $extra --volume=$PWD/model_cache:/model_cache:rw -v $d2repo:/home/appuser/detectron2_repo d2
