extra=""

parentdir="$(dirname "$PWD")"
extra+=" -v /dataset:/dataset"
extra+=" -v /model_zoo:/model_zoo"
extra+=" --volume=$PWD/model_cache:/model_cache:rw"
extra+=" -v $parentdir:/home/appuser/detectron2_repo"

USER_ID=$UID docker-compose run --rm $extra d2
