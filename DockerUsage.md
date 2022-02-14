
HELP
====

* `-d`: runs in background?


* `-p 8888:8888`



# Docker Image

The Docker image for TINerator is based off of the [`jupyter/base-notebook`](https://hub.docker.com/r/jupyter/base-notebook/) image.

## Runtime Options

The TINerator container respects [the Jupyter Docker options](https://jupyter-docker-stacks.readthedocs.io/en/latest/using/common.html#docker-options).

You may customize the execution of the Docker container and the Notebook server it contains with the following optional arguments.

* `-e PASSWORD="YOURPASS"` - Configures Jupyter Notebook to require the given password. Should be conbined with `USE_HTTPS` on untrusted networks.
* `-e USE_HTTPS=yes` - Configures Jupyter Notebook to accept encrypted HTTPS connections. If a `pem` file containing a SSL certificate and key is not found in `/home/jovyan/.ipython/profile_default/security/notebook.pem`, the container will generate a self-signed certificate for you.
* `-e GRANT_SUDO=yes` - Gives the `jovyan` user passwordless `sudo` capability. Useful for installing OS packages. **You should only enable `sudo` if you trust the user or if the container is running on an isolated host.**
* `-v /some/host/folder/for/work:/home/jovyan/work` - Host mounts the default working directory on the host to preserve work even when the container is destroyed and recreated (e.g., during an upgrade).
* `-v /some/host/folder/for/server.pem:/home/jovyan/.ipython/profile_default/security/notebook.pem` - Mounts a SSL certificate plus key for `USE_HTTPS`. Useful if you have a real certificate for the domain under which you are running the Notebook server.
* `-e INTERFACE=10.10.10.10` - Configures Jupyter Notebook to listen on the given interface. Defaults to '*', all interfaces, which is appropriate when running using default bridged Docker networking. When using Docker's `--net=host`, you may wish to use this option to specify a particular network interface.
* `-e PORT=8888` - Configures Jupyter Notebook to listen on the given port. Defaults to 8888, which is the port exposed within the Dockerfile for the image. When using Docker's `--net=host`, you may wish to use this option to specify a particular port.

### Using `sudo`

[d](https://jupyter-docker-stacks.readthedocs.io/en/latest/using/recipes.html#using-sudo-within-a-container)

```bash
docker run -it -e GRANT_SUDO=yes --user root jupyter/minimal-notebook
```
