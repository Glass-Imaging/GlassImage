# Local paths
BUILD_DIR="/Users/manuel/workspace/GlassImage/build-android"
EXE_PATHS=(build-android/tests/unit/*Test)
RUN_PATH="/data/local/tmp/GlassImage"

# Push device
adb push "${BUILD_DIR}" "${RUN_PATH}/"

# Run
CHANGE_DIR="cd ${RUN_PATH}"

for EXE_PATH in "${EXE_PATHS[@]}"
do
    adb shell "${CHANGE_DIR} && ./${EXE_PATH}" || { echo "Test failed, exiting." ; exit 1; }
done

printf "\n\tSCRIPT DONE."