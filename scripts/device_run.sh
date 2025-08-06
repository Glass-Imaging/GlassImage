# Local paths
EXE_FILE="/Users/manuel/workspace/GlassImage/build-android/src/GlassImageTest"
# LIB_FILE="/Users/manuel/workspace/GlassLibrary/build/customers/Motorola/Razr2025/lib/src/libglass_motorola.so"
EXE_NAME=$(basename ${EXE_FILE})


# Device paths
RUN_PATH="/data/local/tmp/GlassImage"


# Push device
adb push "${EXE_FILE}" "${RUN_PATH}/"
# adb push "${LIB_FILE}" "${RUN_PATH}/aarch64-android-qnn229"

# Run
CHANGE_DIR="cd ${RUN_PATH}"
# SET_LIB="export LD_LIBRARY_PATH=${RUN_PATH}/aarch64-android-qnn229"
# SET_DSP="export DSP_LIBRARY_PATH=${RUN_PATH}/aarch64-android-qnn229/unsigned"
# ECHO_PWD='echo Path: $(pwd)'
# RM_OUTPUT="rm -rf ${RUN_PATH}/output_images/*"
# SET_CHMOD="chmod u+x ${EXE_NAME}"
RUN_COMMAND="./${EXE_NAME}"

adb shell "${CHANGE_DIR} && ${RUN_COMMAND}" # || { echo "Run failed, exiting." ; exit 1; }

# # Pull
# rm -rf output_images
# adb pull "${RUN_PATH}/output_images" ./
# adb pull "${RUN_PATH}/trace.json" ./traces/ 


printf "\n\tSCRIPT DONE."