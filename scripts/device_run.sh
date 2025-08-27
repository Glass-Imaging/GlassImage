# Local paths
EXE_FILE="/Users/manuel/workspace/GlassImage/build-android/src/GlassImageTest"
EXE_NAME=$(basename ${EXE_FILE})

# Device paths
RUN_PATH="/data/local/tmp/GlassImage"

# Push device
adb push "${EXE_FILE}" "${RUN_PATH}/"

# Run
CHANGE_DIR="cd ${RUN_PATH}"
RUN_COMMAND="./${EXE_NAME}"

adb shell "${CHANGE_DIR} && ${RUN_COMMAND}" # || { echo "Run failed, exiting." ; exit 1; }

printf "\n\tSCRIPT DONE."