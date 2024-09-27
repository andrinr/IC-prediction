#!/bin/bash

# Check if a directory path is provided as an argument
if [ -z "$1" ]; then
  echo "Usage: $0 <directory-path>"
  exit 1
fi

# List all folders in the specified directory
for dir in "$1"/*/; do
  [ -d "$dir" ] || continue # Make sure it's a directory
  
  # Count the number of items in the directory
  item_count=$(find "$dir" -mindepth 1 -maxdepth 1 | wc -l)
  
  # Display the directory name and item count
  echo "Directory: ${dir%/} - Items: $item_count"
done