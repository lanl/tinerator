## GDAL

### macOS
#### Option 1

[Recommended to install with conda](https://hackernoon.com/install-python-gdal-using-conda-on-mac-8f320ca36d90).


```sh
$ brew install gdal
```

#### Option 2

```sh
```

### Ubuntu
#### Option 1

```sh
$ sudo apt-get install libgdal-dev
```

#### Option 2

```sh
$ sudo apt-add-repository ppa:ubuntugis/ubuntugis-unstable
$ sudo apt-get update
$ sudo apt-get install python-gdal
```

-------

```sh
$ export C_INCLUDE_PATH="$(gdal-config --prefix)/include"
$ export CPLUS_INCLUDE_PATH="$(gdal-config --prefix)/include"

$ pip install GDAL==`gdal-config --version`
```

Optionally,

```sh
$ export LD_LIBRARY_PATH="$(gdal-config --libs):${LD_LIBRARY_PATH}"
$ export CFLAGS="$(gdal-config --cflags):${CFLAGS}"
```

```sh
$ pip install \
    --global-option=build_ext \
    --global-option="$(gdal-config --cflags)" \
    "GDAL==$(gdal-config --version)"
```

You may have luck using [pygdal](#) instead:

```sh
$ python -m pip install "pygdal~=$(gdal-config --version)"
```

Find a subversion from:
- `https://pypi.org/project/pygdal/#history`

## Building and testing
### Building

To build, run:

```sh
$ poetry shell      # create an isolated venv
$ poetry install    # install deps & package
```

### Testing

```sh
$ poetry run pytest
```

### Adding dependencies

```sh
$ poetry add "package==1.2.3"
$ poetry add --optional "package==1.2.3"
$ poetry add --dev "package==1.2.3"
```

### Saving dependencies

```sh
$ poetry lock
```

```sh
$ poetry export \
    --format "requirements.txt" \
    --output "requirements.txt" \
    --without-hashes
```

# Deploying

```sh
$ poetry build
$ poetry publish
```


# Local Testing

```sh
$ act --graph --container-architecture linux/amd64
 ╭─────────────╮ ╭─────────╮ ╭────────╮ ╭──────╮
 │ build-linux │ │ codecov │ │ docker │ │ test │
 ╰─────────────╯ ╰─────────╯ ╰────────╯ ╰──────╯
```

```sh
# Command structure:
act [<event>] [options]
If no event name passed, will default to "on: push"

# List the actions for the default event:
act -l

# List the actions for a specific event:
act workflow_dispatch -l

# Run the default (`push`) event:
act

# Run a specific event:
act pull_request

# Run a specific job:
act -j test

# Run in dry-run mode:
act -n

# Enable verbose-logging (can be used with any of the above commands)
act -v
```



```sh
$ docker build --platform linux/x86_64 -f ci/Dockerfile -t ees16/tinerator:dev ./
```