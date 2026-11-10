#!/bin/bash
set -xvEe -o pipefail

sudo -u svcnbu-swx-hpcx cp  /custom_home/svcnbu-swx-hpcx/.ssh/id_rsa /tmp/id_rsa
sudo chown $(id -u):$(id -g) /tmp/id_rsa
mkdir -p ~/.ssh
chown $(id -u):$(id -g) ~/.ssh
chmod 700 ~/.ssh
mv /tmp/id_rsa ~/.ssh/id_rsa
chmod 600 ~/.ssh/id_rsa
