#!/bin/bash

# Check if the file name argument is provided
if [ -z "$1" ]; then
  echo "Usage: $0 <file_name>"
  exit 1
fi

# File name provided as an argument
FILE_NAME=$1

# Source and destination paths
REMOTE_PATH="shall@data-mover01.hpc.swin.edu.au:/fred/oz395/DeepJSCC/models/saved_models/${FILE_NAME}.h5"
REMOTE_LOGS_PATH="shall@data-mover01.hpc.swin.edu.au:/fred/oz395/DeepJSCC/logs/${FILE_NAME}/"
LOCAL_PATH="models/saved_models/"
LOCAL_PATH_LOGS="logs/${FILE_NAME}/"

# Sync the .h5 file
rsync -avPxH --no-g --chmod=Dg+s "$REMOTE_PATH" "$LOCAL_PATH"

# Sync the logs directory
rsync -avPxH --no-g --chmod=Dg+s "$REMOTE_LOGS_PATH" "$LOCAL_PATH_LOGS"

# Print success messages
echo "File '$FILE_NAME' has been synced to '$LOCAL_PATH'."
echo "Logs directory '$FILE_NAME' has been synced to '$LOCAL_PATH_LOGS'."
