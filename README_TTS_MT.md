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

### 1.2. Install Required Locales

To handle UTF-8 encoding for Maltese characters:

```bash
apt-get install locales
locale-gen en_US.UTF-8
update-locale LC_ALL=en_US.UTF-8 LANG=en_US.UTF-8
```

### 1.3. Install Vim

Install Vim as a text editor, which may be helpful for editing configuration files:

```bash
apt-get install vim
```

### 1.4. Install Sox

Sox is required for audio processing. Install it using:

```bash
apt-get install sox
```

### 1.5. Install Kaldi

To install Kaldi, clone the Kaldi repository, and follow the installation steps:

```bash
# Clone the Kaldi repository
git clone https://github.com/kaldi-asr/kaldi.git /workspace/espnet/tools/kaldi

# Navigate to the Kaldi tools directory and run the installation script
cd /workspace/espnet/tools/kaldi/tools
extras/check_dependencies.sh  # Check for required dependencies
make -j $(nproc)  # Install Kaldi tools

# Navigate to the Kaldi source directory to build Kaldi
cd ../src
./configure --shared
make depend -j $(nproc)
make -j $(nproc)
```

Ensure the Kaldi binaries are in your path:

```bash
export PATH=$PATH:/workspace/espnet/tools/kaldi/src/featbin
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/workspace/espnet/tools/kaldi/src/lib
```

After installation, run the following command to ensure that wav-to-duration - a required tool using during speaker embedding extraction - is properly linked:

```bash
ldd /workspace/espnet/tools/kaldi/src/featbin/wav-to-duration
```

### 1.6. Install Python Requirements

Clone the ESPnet repository and install Python requirements.

#### Clone ESPnet Repository and Pull Latest Changes

```bash
# Clone the forked ESPnet repository
git clone https://github.com/Diskors-AI/espnet.git
cd espnet

# Pull the latest changes from the main ESPnet repository
git remote add upstream https://github.com/espnet/espnet.git
git fetch upstream
git merge upstream/main
```

#### Install ESPnet Dependencies

```bash
# Create a virtual environment and activate it
pyenv virtualenv espnet
pyenv activate espnet

# Initialize submodules
git submodule update --init

# Install ESPnet and required packages
pip install -e ".[tts]" # Install TTS dependencies
pip install kaldiio torchaudio praatio sox tensorboard matplotlib
```

### 1.7. Symlink Directories

Link the data, dump, and exp directories in the ESPnet setup:

```bash
ln -s /workspace/data /workspace/espnet/egs2/vctk/tts1/data
ln -s /workspace/dump /workspace/espnet/egs2/vctk/tts1/dump
ln -s /workspace/exp /workspace/espnet/egs2/vctk/tts1/exp
```

### 1.8. Symlink for split_data.sh

ESPnet expects `split_data.sh` in the `utils/` directory, so create a symbolic link from `utils/data/split_data.sh` to `utils/split_data.sh`:

```bash
ln -s /workspace/espnet/utils/data/split_data.sh /workspace/espnet/utils/split_data.sh
```

### 1.9 Installation of Cython

Run this command:

```bash
cd /workspace/espnet/espnet2/gan_tts/vits/monotonic_align; python setup.py build_ext --inplace
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

## 3. Setting Up VITS with Speaker Embeddings

### 3.1. Enabling Speaker Embeddings and IDs

To leverage the multi-speaker capability of VITS, we’ll enable speaker embeddings with Kaldi for x-vector extraction and ensure each speaker has a unique identifier (SID). Additionally, enabling GST (Global Style Tokens) is optional but beneficial if the dataset includes varied speaking styles.

Run the following command to initiate speaker embedding extraction and set Kaldi as the embedding tool:

```bash
./run.sh --stage 2 --stop-stage 3 --use_spk_embed true --spk_embed_tool kaldi --use_sid true --nj <no_of_cpus>
```

Explanation of flags:

- `--use_spk_embed true`: Enables extraction of speaker embeddings to differentiate voices in multi-speaker datasets.
- `--spk_embed_tool kaldi`: Specifies Kaldi as the tool for x-vector embedding extraction, utilizing pre-trained x-vector embeddings.
- `--use_sid true`: Assigns a unique Speaker ID (SID) to each speaker, helping the model distinguish between them.

## 4. Training the VITS Model

Run the VITS training steps with speaker embeddings and the pre-trained model configuration:

```bash
nohup ./run.sh --stage 4 --stop-stage 7 \
    --tag vits_xvector_maltese \
    --nj <no_of_cpus> \
    --ngpu <gpus_count> > nohup.out 2>&1 &
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
