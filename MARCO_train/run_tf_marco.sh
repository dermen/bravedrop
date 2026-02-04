#!/bin/bash

# Path to the file containing the list of .png files
#file_list=/data/brave/dermen/xtals2.txt
file_list=$1
# This will indicate what file it's "reading"
echo "Getting images from $file_list"
#Does this need to be in  "" or can it be without them?

# Path to the Python 
python_script="/home/sw/deb/x86_64/phenix/phenix-1.21.1-5286/build/bin/libtbx.python"
python_script_path="/data/brave/MARCO/MS/savedmodelfolder/jpeg2json2.py"

# Folder to save the .jpeg files
output_folder="jpeg_files"

# Output JSON file
output_json=all_files.json

# Initialize the output JSON file
rm $output_json

# Create the output folder if it doesn't exist
mkdir -p "$output_folder"

# Initialize the counter
counter=1

# Read each line from the file
while IFS= read -r png_file; do
  # Check if the file exists and is a .png file
  if [[ -f "$png_file" && "$png_file" == *.png ]]; then
    # Determine the name for the output .jpeg file
    jpeg_file=""
    case $counter in
      1) jpeg_file="file_one.jpeg" ;;
      2) jpeg_file="file_two.jpeg" ;;
      3) jpeg_file="file_three.jpeg" ;;
      4) jpeg_file="file_four.jpeg" ;;
      5) jpeg_file="file_five.jpeg" ;;
      # Add more cases as needed
      *) jpeg_file="file_${counter}.jpeg" ;;
    esac
    jpeg_file="$output_folder/$jpeg_file"
    
    # Convert the .png file to .jpeg using the convert command from ImageMagick
    convert "$png_file" "$jpeg_file"
    
    echo "Converted $png_file to $jpeg_file"

    # Convert the .jpeg file to JSON using the Python script and append to the output JSON file
    $python_script $python_script_path "$jpeg_file" >> $output_json
    
    # Remove trailing comma from previous JSON object and add current JSON object
    #sed -i '$ s/,$//' "$output_json"
    #echo "$json_output," >> "$output_json"

    echo "Converted $jpeg_file to JSON and appended to $output_json"

    # Increment the counter
    counter=$((counter + 1))
  else
    echo "File $png_file does not exist or is not a .png file"
  fi
done < "$file_list"

# Remove the trailing comma from the last JSON object and close the JSON array
#sed -i '$ s/,$//' "$output_json"
#echo "]" >> "$output_json"


#/home/sw/deb/x86_64/phenix/phenix-1.21.1-5286/build/bin/libtbx.python
#    jpeg2json2.py filesinjpseg.jpeg > list_of_files.json
##CHECK PYTHON2 AND PYTHON 3 ISSUE!!!

#gcloud whatever ... .. list_of_files.json
#theobe below should work (hopefully!!!)

gcloud ml-engine local predict --model-dir=savedmodel --json-instances=$output_json
#All of the output json files are all_files.json

#|tee new_500.log
