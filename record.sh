#!/usr/bin/env bash
set -e

if [ "$#" -lt 2 ]; then
  echo "Usage: $0 <output.mp4> <command> [args...]"
  exit 1
fi

OUTPUT="$1"
shift
CMD=("$@")

DISPLAY_NUM=:99
# Adapt resolution to cmd content
if [[ "${CMD[*]}" == *tetris* ]]; then
  RESOLUTION=384x928
else
  RESOLUTION=1280x720
fi
FRAMERATE=30

cleanup() {
  if kill -0 "$FFMPEG_PID" 2>/dev/null; then
    kill -INT "$FFMPEG_PID"
    wait "$FFMPEG_PID" 2>/dev/null || true
  fi
  kill "$Xvfb_PID" 2>/dev/null || true
}
trap cleanup EXIT

Xvfb $DISPLAY_NUM -screen 0 ${RESOLUTION}x24 &
Xvfb_PID=$!

export DISPLAY=$DISPLAY_NUM

command -v openbox >/dev/null && openbox &

ffmpeg -y \
  -video_size $RESOLUTION \
  -framerate $FRAMERATE \
  -f x11grab \
  -draw_mouse 0 \
  -i ${DISPLAY_NUM}.0 \
  -preset ultrafast \
  "$OUTPUT" &
FFMPEG_PID=$!

"${CMD[@]}"