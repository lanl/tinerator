#!/bin/bash

# This script will format module code using the 
# Black code formatter. 
# Line length is 79 chars as outlined in PEP 8.
# This script should be run before every commit.

find ../tinerator -name "*.py" | xargs black -l 79 -t py38
