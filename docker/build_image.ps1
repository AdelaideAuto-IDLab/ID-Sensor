$env:DOCKER_TLS_VERIFY="1"
$env:DOCKER_HOST="tcp://management-1.ambigem.local:2376"
$env:DOCKER_CERT_PATH="$HOME/.docker/management-1.ambigem.local"

docker build --network=host -t deeplearning-training .