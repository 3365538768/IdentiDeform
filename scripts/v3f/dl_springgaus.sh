#!/usr/bin/env bash
# V3f: download Spring-Gaus released data (Google Drive) with retries into data/external_datasets/springgaus_gdrive (untrusted).
source /kairos-yangsen/kairis-vepfs-6t/linrui/env.sh
R=/kairos-yangsen/kairis-vepfs-6t/linrui/rsi_real_test
D=$R/data/external_datasets/springgaus_gdrive
cd /tmp
for pair in "1HKYOPG9gNDSmfLkhUo5W0Lw9sM8LGGbm mpm_synthetic.zip" "10u5_gnKh6pdZMNtysPMmv8AKd6KjYoXL real_capture.zip"; do
  set -- $pair
  for a in $(seq 1 40); do
    $R/.engine-envs/v3f_dl/bin/python -I -c "import gdown,sys; gdown.download(id=sys.argv[1], output=sys.argv[2], quiet=True, resume=True)" "$1" "$D/$2" && unzip -tq "$D/$2" >/dev/null 2>&1 && { echo "$(date -Is) $2 ok $(stat -c %s $D/$2)"; break; }
    echo "$(date -Is) $2 attempt $a failed"; sleep 20
  done
done
echo ALLDONE
