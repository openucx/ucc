#!/bin/bash

set -xvEe -o pipefail

mkdir -p ~/.config/enroot
if [ ! -s ~/.config/enroot/.credentials ]; then
    echo "INFO: Create enroot credentials file with some content"
    echo "this mitigates error reporting for anonimous image pull"
    echo "# This comment is to mitigate Enroot credentials file missing error" > ~/.config/enroot/.credentials
fi
