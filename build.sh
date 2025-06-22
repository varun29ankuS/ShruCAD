#!/usr/bin/env bash
# build.sh

# Exit on error
set -o errexit

# These commands are run by Render before it installs python packages.
echo "---> Installing Tesseract OCR..."
apt-get update
apt-get install -y tesseract-ocr
