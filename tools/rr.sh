#!/bin/sh
# あたらしい作りで 必殺技動画を 作りなおす。使い方: rr.sh <0|1>  (2本同時に回す)
k=$1; n=0; mkdir -p /tmp/hsw$k
for i in $(cat /home/claude/hs3/order.txt); do
  n=$((n+1)); [ $((n%2)) -eq $k ] || continue
  [ -f /home/claude/hs3/$i.mp4 ] && continue
  rm -f /tmp/hsw$k/*; python3 /home/claude/tools/hs_lib.py $i /tmp/hsw$k >/dev/null 2>&1 && mv /tmp/hsw$k/$i.mp4 /home/claude/hs3/$i.mp4
done
