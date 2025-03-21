#!/bin/bash

if [ "$#" -ne 1 ]; then
    echo "Usage: $0 <filename>"
    exit 1
fi

FILENAME=$1
ABSOLUTE_PATH=$(realpath "$FILENAME")
ABSOLUTE_DIR=$(dirname "$ABSOLUTE_PATH")
OUTPUT_FOLDER="${ABSOLUTE_DIR}/trimmed"
PROCESSED_FILENAME="${FILENAME%.*}_processed.${FILENAME##*.}"
TRIMMED_FILENAME="${FILENAME%.*}_trimmed.${FILENAME##*.}"

echo "$(date -u +"%Y-%m-%dT%H:%M:%SZ")"

PROCESSED_BASENAME=$(basename "${TRIMMED_FILENAME}")
echo "Checking if file exists: ${OUTPUT_FOLDER}/${PROCESSED_BASENAME}"
if [ -f "${OUTPUT_FOLDER}/${PROCESSED_BASENAME}" ]; then
    echo "File already exists: ${OUTPUT_FOLDER}/${PROCESSED_BASENAME}"
    exit 1
fi

echo "Making ${OUTPUT_FOLDER}"
mkdir -p ${OUTPUT_FOLDER}

echo "Processing ${FILENAME}"
python main.py --input "${FILENAME}" --output "${PROCESSED_FILENAME}" --model_size m --skip_frames 3 --smoothing_window 30 --conf_threshold 0.5 --use_saliency --max_workers 6 --target_ratio 0.75

# echo "Trimming ${PROCESSED_FILENAME}"
# ffmpeg -i ${PROCESSED_FILENAME} -ss 5 -to $(ffmpeg -i ${PROCESSED_FILENAME} 2>&1 | awk -F: '/Duration/ {print $2*3600 + $3*60 + $4 - 5.5}') -c copy ${TRIMMED_FILENAME}

echo "Moving ${PROCESSED_FILENAME} to ${OUTPUT_FOLDER}"
mv ${PROCESSED_FILENAME} ${OUTPUT_FOLDER}

# echo "Removing ${PROCESSED_FILENAME}"
# rm ${PROCESSED_FILENAME}

echo "$(date -u +"%Y-%m-%dT%H:%M:%SZ")"
