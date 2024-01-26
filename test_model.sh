#! /usr/bin/env bash

# readonly DEVICE="$1"
# shift
DEVICE="vulkan://0"

TARGET_FLAGS=()
if [[ $DEVICE =~ "vulkan" ]]; then
  TARGET_FLAGS=("--iree-hal-target-backends=vulkan-spirv" "--iree-vulkan-target-triple=rdna3-7900-linux")
elif [[ $DEVICE =~ "rocm" ]]; then
  TARGET_FLAGS=("--iree-hal-target-backends=rocm" "--iree-rocm-target-chip=gfx940")
fi

list="./model_list.txt"
while IFS= read -r HF_ID
do
  echo "$HF_ID"
  NAME=${HF_ID//['-''/']/'_'}

  python python/turbine_models/model_builder.py "${HF_ID}" > tmp.txt
  if [[ ! "$(cat tmp.txt)" =~ "done" ]]; then
    echo "${HF_ID} Fail" >> results.txt
    continue
  fi

  iree-compile "${NAME}".mlir -o "${NAME}".vmfb "${TARGET_FLAGS[@]}" > tmp.txt
  if [ -s tmp.txt ]; then
    echo "${HF_ID} Fail" >> results.txt
    continue
  fi

  iree-benchmark-module --module="${NAME}".vmfb --device="${DEVICE}" --function=run --input=1x1xi64 > tmp.txt
  if [[ "$(cat tmp.txt)" =~ "items_per_second" ]]; then
    echo "${HF_ID} Pass" >> results.txt
  else
    echo "${HF_ID} Fail" >> results.txt
  fi
done < "$list"
