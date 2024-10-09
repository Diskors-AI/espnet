# TTS Training Workflow for VITS

This flow provides step-by-step instructions for setting up and training a Text-to-Speech (TTS) model using ESPnet with the VITS architecture. It assumes that you already have a WAV corpus aligned using the Montreal Forced Aligner (MFA) and that the corresponding TextGrid files have been generated. The aligned data will be used to prepare training data, extract speaker embeddings, and train the VITS TTS model.

## 1. Initial Setup

### 1.1. Directory Structure

Ensure the following directory structure in your workspace:

```
/workspace/
├── espnet/             # ESPnet repo
├── exp/                # Experiment results and trained models
├── dump/               # Processed data (features, etc.)
└── data/               # Raw data (audio + transcription)
    └── corpus/         # Corpus data (WAV files + alignments)
        ├── wav/        # WAV audio files
        └── alignments/ # MFA TextGrid alignments
```

### 1.2. Install Required Locales and Dependencies

Install UTF-8 Locales

To handle Maltese characters:

```bash
apt-get install locales
locale-gen en_US.UTF-8
update-locale LC_ALL=en_US.UTF-8 LANG=en_US.UTF-8
```

#### Install Python Requirements

Clone the forked ESPnet repository, pull the latest changes from the main ESPnet repo, and install the required packages:

```bash
# Clone the forked ESPnet repository
git clone https://github.com/Diskors-AI/espnet.git
cd espnet

# Pull the latest changes from the main ESPnet repository
git remote add upstream https://github.com/espnet/espnet.git
git fetch upstream
git merge upstream/main

# Create a venv if it does not exist yet.
pyenv virtualenv espnet
pyenv activate espnet

# Initialize submodules
git submodule update --init

# Install Python requirements
pip install -e ".[tts]" # Install TTS dependencies
pip install kaldiio torchaudio
```

#### Symlink Directories

Link the data, dump, and exp directories in the ESPnet setup:

```bash
ln -s /workspace/data /workspace/espnet/egs2/vctk/tts1/data
ln -s /workspace/dump /workspace/espnet/egs2/vctk/tts1/dump
ln -s /workspace/exp /workspace/espnet/egs2/vctk/tts1/exp
```

## 2. Data Preparation

### 2.1. Downsampling Audio

Downsample WAV files from 44100 Hz to 22050 Hz if necessary, and convert to Mono, for optimal TTS performance.

### 2.2. Prepare Data from MFA Alignments

**Relative Path**: egs2/vctk/tts1/pyscripts/utils

Use MFA alignments to prepare data for ESPnet. Each TextGrid file should contain three tiers: words, phones, and utterances.

```bash
python mfa_prep_stage2_data.py --alignments_dir /workspace/data/corpus/alignments --corpus_dir /workspace/data/corpus/wav --output_dir /workspace/data
```

### 2.3. Validate Data

**Relative Path**: egs2/vctk/tts1

Validate the prepared data:

```bash
./utils/validate_data_dir.sh --no-feats /workspace/data/tr_no_dev
./utils/validate_data_dir.sh --no-feats /workspace/data/dev
./utils/validate_data_dir.sh --no-feats /workspace/data/eval1
```

### 2.4. Prepare Phoneme Data

Create phoneme data for each dataset (train, dev, eval):

```bash
./utils/mfa_prep_stage2_phoneme_data.sh /workspace/data
```

## 3. Setting Up VITS with Speaker Embeddings

### 3.1. Enabling Speaker Embeddings and IDs

To leverage the multi-speaker capability of VITS, we’ll enable speaker embeddings with Kaldi for x-vector extraction and ensure each speaker has a unique identifier (SID). Additionally, enabling GST (Global Style Tokens) is optional but beneficial if the dataset includes varied speaking styles.

Run the following command to initiate speaker embedding extraction and set Kaldi as the embedding tool:

```bash
./run.sh --stage 2 --stop-stage 3 --use_spk_embed true --spk_embed_tool kaldi --use_sid true --use_gst true
```

Explanation of flags:

- `--use_spk_embed true`: Enables extraction of speaker embeddings to differentiate voices in multi-speaker datasets.
- `--spk_embed_tool kaldi`: Specifies Kaldi as the tool for x-vector embedding extraction, utilizing pre-trained x-vector embeddings.
- `--use_sid true`: Assigns a unique Speaker ID (SID) to each speaker, helping the model distinguish between them.
- `--use_gst true`: (optional) Activates GST to capture style variations, such as prosody and intonation, improving the model’s ability to generalize across different speech styles.

### 3.2. Using the Pre-trained VCTK VITS Model as Starting Point

Download the pre-trained VCTK VITS model to use as an initialization point for training:

```bash
cd egs2/vctk/tts1
wget https://zenodo.org/record/5500759/files/tts_train_multi_spk_vits_raw_phn_tacotron_g2p_en_no_space_train.total_count.ave.zip?download=1 -O pretrained_vctk_vits.zip

# Unzip and place the model in the `exp/` directory
unzip pretrained_vctk_vits.zip -d exp/pretrained_vctk_vits
```

The model's pth file will be accessible at `exp/pretrained_vctk_vits/exp/tts_train_multi_spk_vits_raw_phn_tacotron_g2p_en_no_space/train.total_count.ave_10best.pth`

## 4. Training the VITS Model

Run the VITS training steps with speaker embeddings and the pre-trained model configuration:

```bash
./run.sh --stage 4 --stop-stage 6 \
    --train_config conf/tuning/train_xvector_vits.yaml \
    --use_xvector true \
    --xvector_dir data/train/xvector.scp \
    --train_args "--init_param /path/to/pretrained_model.pth:tts:tts" \
    --tag finetune_vits_xvector_maltese
```

### 4.1. Monitor Training Progress

Check logs for validation and training metrics:

#### Monitor training logs

```bash
tail -f exp/pretrained_vctk_vits/log/train.log
```

## 5. Inference and Evaluation

After training, run inference to test the model:

```bash
./run.sh --stage 7 --stop-stage 8 --inference-config conf/decode.yaml --inference_tag test_with_vits
```
