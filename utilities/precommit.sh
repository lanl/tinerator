#!/bin/bash

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"

echo ${DIR}

generate_docs() {
    echo "Doc generation: not implemented yet"
}

# This script will format module code using the 
# Black code formatter. 
# Line length is 79 chars as outlined in PEP 8.
# This script should be run before every commit.
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
    echo "./format_code [format | docs | all]"
    ;;
esac