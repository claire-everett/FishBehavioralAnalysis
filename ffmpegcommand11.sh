#!/bin/sh
#script to process images

import ffmpeg

counter = 0
for dir in */; do
    for i in "$dir"/*.h264 ; do 
        ffmpeg -y -i "$i" -c copy -f h264 needle.h264
        ffmpeg -y -r 40 -i needle.h264 -c copy $(basename "${i/.h264}").mp4
        ffmpeg -i $(basename "${i/.h264}").mp4 -filter:v "crop=500:500:650:100" $(basename "${i/.h264}")2.mp4  
        echo "file '$(basename "${i/.h264}")2.mp4'" >> output.txt
    done 
    counter=$((counter+1))
done
ffmpeg -f concat -i output.txt -vcodec copy -acodec copy $((counter+1))Final.mp4


cp -r ./ ../../PostProcess/BettaDirectory11/

rm -r  /Users/Claire/Desktop/BettaDirectory11/bettaDirectory
