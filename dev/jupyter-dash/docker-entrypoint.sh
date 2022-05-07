#!/usr/bin/env

exec /home/$CONTAINER_USER/miniconda/envs/tinerator/bin/jupyter lab \
        --ip=0.0.0.0 \
        -y \
        --log-level='INFO' \
        --port=$PORT_JUPYTER