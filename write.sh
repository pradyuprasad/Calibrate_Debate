#!/bin/bash

# Check if a directory is provided, default to current directory if not
directory="${1:-.}"

# Find and display all .py files recursively, excluding dot directories
find "$directory" -type f -name "*.py" -not -path "*/\.*/*" | while read -r file; do
    echo -e "\n========== $file ==========\n"
    cat "$file"
    echo -e "\n================================\n"
done
