#!/bin/bash

# Size of the model to use. Options: "s", "m", "l", "x"
MODEL_SIZE="x"

# Number of frames to skip for saliency detection
SKIP_FRAMES=10

# Number of frames for temporal video smoothing
SMOOTHING_WINDOW=10

# Confidence threshold for saliency detection
# 0.0 to 1.0
CONF_THRESHOLD=0.4

# Number of workers for processing
# This should be set to the number of CPU cores available
MAX_WORKERS=6

# Ratio of the target size to the original size
# 4:3 aspect ratio is 0.75
# 16:9 aspect ratio is 0.5625
# 1:1 aspect ratio is 1.0
TARGET_RATIO=0.5625


# Check if the input file is provided
if [ "$#" -ne 1 ]; then
    echo "Usage: $0 <filename>"
    exit 1
fi

FILENAME=$1
BASE_FILENAME=$(basename "${FILENAME}")
ABSOLUTE_PATH=$(realpath "$FILENAME")
ABSOLUTE_DIR=$(dirname "$ABSOLUTE_PATH")
OUTPUT_FOLDER="${ABSOLUTE_DIR}/portrait"
PROCESSED_FILENAME="${FILENAME%.*}_processed.${FILENAME##*.}"
PROCESSED_BASENAME=$(basename "${PROCESSED_FILENAME}")

echo "$(date -u +"%Y-%m-%dT%H:%M:%SZ")"

echo "Checking if file exists: ${OUTPUT_FOLDER}/${PROCESSED_FILENAME}"
if [ -f "${OUTPUT_FOLDER}/${PROCESSED_BASENAME}" ]; then
    echo "File already exists: ${OUTPUT_FOLDER}/${PROCESSED_BASENAME}"
    exit 1
fi

echo "Making ${OUTPUT_FOLDER}"
mkdir -p ${OUTPUT_FOLDER}

echo "Processing:"
echo --input "${FILENAME}" --output "${PROCESSED_FILENAME}" --model_size ${MODEL_SIZE} --skip_frames ${SKIP_FRAMES} --smoothing_window ${SMOOTHING_WINDOW} --conf_threshold ${CONF_THRESHOLD} ${USE_SALIENCY} --max_workers ${MAX_WORKERS} --target_ratio ${TARGET_RATIO}


python main.py  --input "${FILENAME}" \
                --output "${PROCESSED_FILENAME}" \
                --target_ratio ${TARGET_RATIO} \
                --model_size ${MODEL_SIZE} \
                --skip_frames ${SKIP_FRAMES} \
                --smoothing_window ${SMOOTHING_WINDOW} \
                --conf_threshold ${CONF_THRESHOLD} \
                --max_workers ${MAX_WORKERS}

mv ${PROCESSED_FILENAME} ${OUTPUT_FOLDER}/${BASE_FILENAME}
echo "$(date -u +"%Y-%m-%dT%H:%M:%SZ")"
