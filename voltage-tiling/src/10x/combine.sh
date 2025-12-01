#!/bin/bash



# create funciton to merge geojson files
merge_geojson() {
    local region="$1"

    # define the directory path
    prefix="/Users/ilyonsg/Documents/nrel/data/sfo/"
    # input path is the subdirectory containing the geojson files
    input_dir="base_timeseries/"
    output_dir="base_timeseries_combined/"
    # input path is the subdirectory containing the geojson files
    input_path="$prefix$input_dir$region"
    output_path="$prefix$output_dir"

    # node modules path
    node_modules_path="/usr/local/lib/node_modules"

    # create the output directory
    mkdir -p "$output_path"

    # merge the geojson files
    cd $node_modules_path
    geojson-merge "$input_path"/*.json > "$output_path"/"$region".json
}


# define the directory path
prefix="/Users/ilyonsg/Documents/nrel/data/sfo/"
# input path is the subdirectory containing the geojson files
input_dir="base_timeseries/"
input_path="$prefix$input_dir"

# loop over all the regions in input directory
for region in "$input_path"/*/; do
    region=$(basename "$region")
    merge_geojson "$region"
done