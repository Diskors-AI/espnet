#!/usr/bin/env bash
# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail

# Parameters for audio processing
fs=22050
n_fft=1024
n_shift=256
win_length=1024

# Determine audio format
opts=
if [ "${fs}" -eq 22050 ]; then
    # To suppress recreation, specify wav format
    opts="--audio_format wav "
else
    opts="--audio_format flac "
fi

# Dataset and model configurations
train_set=tr_no_dev
valid_set=dev
test_sets="dev eval1"
g2p=none

# Model configuration
train_config=conf/tuning/train_xvector_vits.yaml
inference_config=conf/tuning/decode_vits.yaml

./tts.sh \
    --lang en \
    --feats_type raw \
    --fs "${fs}" \
    --n_fft "${n_fft}" \
    --n_shift "${n_shift}" \
    --win_length "${win_length}" \
    --token_type phn \
    --cleaner maltese \
    --g2p "${g2p}" \
    --tts_task gan_tts \
    --use_spk_embed true \
    --spk_embed_tool kaldi \
    --spk_embed_tag xvector \
    --train_config "${train_config}" \
    --inference_config "${inference_config}" \
    --train_set "${train_set}" \
    --valid_set "${valid_set}" \
    --test_sets "${test_sets}" \
    --srctexts "data/${train_set}/text" \
    ${opts} "$@"