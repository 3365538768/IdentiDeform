#!/usr/bin/env bash
# V3f PIXIE: download released PIXIE UNet checkpoints (HF dataset vlongle/pixie, only checkpoints_*) and the
# OpenAI CLIP ViT-L/14@336px weights f3rm uses (sha256-verified). Retries for the flaky network.
source /kairos-yangsen/kairis-vepfs-6t/linrui/env.sh
R=/kairos-yangsen/kairis-vepfs-6t/linrui/rsi_real_test
cd /tmp
for a in $(seq 1 30); do
  $R/.venv/bin/python -I $R/scripts/v3f/pixie_hf_download.py $R/external/pixie_hf && break
  echo "$(date -Is) hf attempt $a failed"; sleep 20
done
SHA=3035c92b350959924f9f00213499208652fc7ea050643e8b385c2dac08641f02
F=$R/external/clip_vitl14_336/ViT-L-14-336px.pt
for a in $(seq 1 40); do
  curl -sSL -C - --retry 5 -o $F https://openaipublic.azureedge.net/clip/models/$SHA/ViT-L-14-336px.pt
  if [ "$(sha256sum $F | cut -d' ' -f1)" = "$SHA" ]; then echo "$(date -Is) clip ok"; break; fi
  echo "$(date -Is) clip attempt $a: size $(stat -c %s $F 2>/dev/null)"; sleep 15
done
echo ALLDONE
