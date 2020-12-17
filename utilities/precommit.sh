#!/bin/bash

# This script should be called before every commit
# with the 'all' option.
# Prepares and generates documentation using MkDocs
# and formats Python code using Black.

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"

# Generate documentation
generate_docs() {
    echo "Doc generation: not implemented yet"
    # https://pydoc-markdown.readthedocs.io/en/latest/docs/renderers/mkdocs/
}

# Format code via Black
format_code() {
    find ${DIR}/../tinerator -name "*.py" | xargs black -l 79 -t py38
}

case $1 in

  "format")
    format_code
    ;;

  "docs")
    generate_docs
    ;;

  "all")
    format_code
    generate_docs
    ;;

  *)
    echo "./precommit.sh [format | docs | all]"
    ;;
esac