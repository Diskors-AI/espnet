# TTS Training Workflow for Tacotron2 and FastSpeech2

This flow provides step-by-step instructions for setting up and training a Text-to-Speech (TTS) model using ESPnet. It assumes that you already have a WAV corpus that has been force-aligned using the Montreal Forced Aligner (MFA) and that the corresponding TextGrid files have been generated. The aligned data will be used in subsequent stages to prepare training data, extract speaker embeddings, and train TTS models such as Tacotron2 and FastSpeech2.

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

#### Install UTF-8 Locales

This ensures Maltese characters are processed correctly:

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

# Initialize submodules
git submodule update --init

# Install Python requirements
pip install -e .

# Install additional required packages
pip install praatio tensorboard matplotlib torchaudio wheel flash_attn
pip install -U espnet_model_zoo
```

#### Symlink Directories

Link the data, dump, and exp directories in the ESPnet setup:

```bash
ln -s /workspace/data /workspace/espnet/egs2/ljspeech/tts1/data
ln -s /workspace/dump /workspace/espnet/egs2/ljspeech/tts1/dump
ln -s /workspace/exp /workspace/espnet/egs2/ljspeech/tts1/exp
```

## 2. Data Preparation

### 2.1. Downsampling Audio

Downsample WAV files from 44100 Hz to 22050 Hz since it's sufficient for voice compression.

### 2.2. Prepare Data from MFA Alignments

**Relative Path**: `egs2/ljspeech/tts1/pyscripts/utils`

Prepare data for ESPnet from Montreal Forced Aligner (MFA) alignments. Each TextGrid file should contain three tiers: words, phones, and utterances.

#### Command:

```bash
python mfa_prep_stage2_data.py --alignments_dir /workspace/data/corpus/alignments \
    --corpus_dir /workspace/data/corpus/wav \
    --output_dir /workspace/data
```

### 2.3. Validate Data

**Relative Path**: `egs2/ljspeech/tts1`

Validate the prepared data to ensure correct formatting.

```bash
./utils/validate_data_dir.sh --no-feats /workspace/data/tr_no_dev
./utils/validate_data_dir.sh --no-feats /workspace/data/dev
./utils/validate_data_dir.sh --no-feats /workspace/data/eval1
```

### 2.4. Prepare Phoneme Data

**Relative Path**: `egs2/ljspeech/tts1`

Create equivalent phoneme data for each dataset (train, dev, eval):

```bash
./utils/mfa_prep_stage2_phoneme_data.sh /workspace/data
```

## 3. Tacotron2 Training

### 3.1. Run Stages 2 and 3: Data Preparation and Speaker Embedding Extraction

**Relative Path**: `egs2/ljspeech/tts1`

Enable speaker embedding extraction by setting `--use_spk_embed true` and specify the embedding tool (e.g., `espnet`):

```bash
./run.sh --stage 2 --stop-stage 3 --use_spk_embed true --spk_embed_tool espnet
```

### 3.2. (Optional) Run Stages 4-5: Filter Utterances and Generate Token List

**Relative Path**: `egs2/ljspeech/tts1`

Run this step if filtering utterances and generating token lists are necessary:

```bash
./run.sh --stage 4 --stop-stage 5 --use_spk_embed true
```

### 3.3. Run Stages 6-8: Train the Tacotron2 Teacher Model

**Relative Path**: `egs2/ljspeech/tts1`

Train the teacher model (Tacotron2) using the prepared data:

```bash
./run.sh --stage 6 --use_spk_embed true
```

## 4. FastSpeech2 Training

After completing Tacotron2 training, follow these steps to train FastSpeech2.

### 4.1. Generate Durations with Teacher-Forcing Mode

**Relative Path**: `egs2/ljspeech/tts1`

Decode the data using teacher-forcing mode to get groundtruth-aligned durations.

```bash
./run.sh --stage 8 \
    --tts_exp exp/tts_train_raw_phn_tacotron_g2p_en_no_space \
    --inference_args "--use_teacher_forcing true" \
    --test_sets "tr_no_dev dev eval1"
```

This will generate durations in:

```
exp/tts_train_raw_phn_tacotron_g2p_en_no_space/decode_use_teacher_forcingtrue_train.loss.ave
```

### 4.2. Calculate Statistics for FastSpeech2 Training

**Relative Path**: `egs2/ljspeech/tts1`

You need to calculate additional statistics like F0 and energy for FastSpeech2. Use the following command:

```bash
./run.sh --stage 6 \
    --train_config conf/tuning/train_fastspeech2.yaml \
    --teacher_dumpdir exp/tts_train_raw_phn_tacotron_g2p_en_no_space/decode_use_teacher_forcingtrue_train.loss.ave \
    --tts_stats_dir exp/tts_train_raw_phn_tacotron_g2p_en_no_space/decode_use_teacher_forcingtrue_train.loss.ave/stats \
    --write_collected_feats true
```

### 4.3. Train FastSpeech2

**Relative Path**: `egs2/ljspeech/tts1`

Finally, train FastSpeech2 using the durations generated from the teacher model:

```bash
./run.sh --stage 7 \
    --train_config conf/tuning/train_fastspeech2.yaml \
    --teacher_dumpdir exp/tts_train_raw_phn_tacotron_g2p_en_no_space/decode_use_teacher_forcingtrue_train.loss.ave
```
