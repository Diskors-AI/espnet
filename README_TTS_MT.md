# TTS Training Workflow for VITS

This guide provides a step-by-step workflow for setting up and training a Text-to-Speech (TTS) model using ESPnet’s VITS architecture. It assumes you have already aligned your WAV corpus using the Montreal Forced Aligner (MFA) and have generated corresponding TextGrid files.

The process includes:

1. Setting up the environment and installing dependencies.
2. Preparing data from MFA alignments.
3. Extracting speaker embeddings for multi-speaker scenarios.
4. Training the VITS model.
5. Performing inference and evaluation.

This workflow enables you to experiment with different configurations while reusing previously prepared data and speaker embeddings, thus allowing rapid iteration without rerunning all preprocessing steps.

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

If your language involves special characters (e.g., Maltese), install and configure locales:

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

### 1.5. Install Python Requirements

Clone the ESPnet repository, set up a Python environment, and install dependencies.

```bash
cd /workspace
git clone https://github.com/Diskors-AI/espnet.git
cd espnet

# Update from main ESPnet repository
git remote add upstream https://github.com/espnet/espnet.git
git fetch upstream
git merge upstream/master

# Setup Python environment (example using pyenv)
pyenv virtualenv espnet
pyenv activate espnet

git submodule update --init
pip install -e ".[tts]"
pip install kaldiio torchaudio praatio sox tensorboard matplotlib
pip install wheel
pip install flash_attn
```

### 1.6. Install Kaldi

Install Kaldi inside the ESPnet tools directory:

```bash
# Clone the Kaldi repository
git clone https://github.com/kaldi-asr/kaldi.git /workspace/espnet/tools/kaldi

# Navigate to the Kaldi tools directory and run the installation script
cd /workspace/espnet/tools/kaldi/tools
extras/check_dependencies.sh  # Check for required dependencies
make -j $(nproc)  # Install Kaldi tools
extras/install_openblas.sh # Install OpenBlas for Non-Intel CPUs

# Navigate to the Kaldi source directory to build Kaldi
cd ../src
./configure --shared # For Intel CPUs
or
./configure --shared --mathlib=OPENBLAS # For non-Intel CPUs
make depend -j $(nproc)
make -j $(nproc)
```

Add Kaldi binaries to your PATH:

```bash
export PATH=$PATH:/workspace/espnet/tools/kaldi/src/featbin
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/workspace/espnet/tools/kaldi/src/lib
```

Verify wav-to-duration is correctly linked:

```bash
ldd /workspace/espnet/tools/kaldi/src/featbin/wav-to-duration
```

### 1.7. Symlink Data, Dump, and Exp Directories

Link your main data directories into the ESPnet example recipe directory (here, using vctk/tts1 as an example):

```bash
ln -s /workspace/data /workspace/espnet/egs2/vctk/tts1/data
ln -s /workspace/dump /workspace/espnet/egs2/vctk/tts1/dump
ln -s /workspace/exp /workspace/espnet/egs2/vctk/tts1/exp
```

### 1.8. Symlink for split_data.sh

ESPnet expects `split_data.sh` in `utils/`:

```bash
# This link did not work last time
ln -s /workspace/espnet/utils/data/split_data.sh /workspace/espnet/utils/split_data.sh
# I used this instead.
ln -s /workspace/espnet/egs2/TEMPLATE/asr1/utils/data/split_data.sh utils/split_data.sh
```

### 1.9 Install Cython Extensions for VITS

Build the monotonic alignment extension for VITS:

```bash
cd /workspace/espnet/espnet2/gan_tts/vits/monotonic_align; python setup.py build_ext --inplace
```

## 2. Data Preparation

### 2.1. Downsampling Audio

If your audio is not already at 22050 Hz mono, downsample and convert it before proceeding.

### 2.2. Prepare Data from MFA Alignments

**Relative Path**: egs2/vctk/tts1/pyscripts/utils

Use the MFA-produced TextGrid alignments to generate training data lists and transcription files. Each TextGrid file should contain three tiers: words, phones, and utterances:

```bash
python mfa_prep_stage2_data.py \
    --alignments_dir /workspace/data/corpus/alignments \
    --corpus_dir /workspace/data/corpus/wav \
    --output_dir /workspace/data \
    --data_percentage 1.0
```

### 2.3. Validate Data

**Relative Path**: egs2/vctk/tts1

Run validation checks to ensure data directories are correct:

```bash
./utils/validate_data_dir.sh --no-feats /workspace/data/tr_no_dev
./utils/validate_data_dir.sh --no-feats /workspace/data/dev
./utils/validate_data_dir.sh --no-feats /workspace/data/eval1
./utils/validate_data_dir.sh --no-feats /workspace/data/tr_no_dev_phn
./utils/validate_data_dir.sh --no-feats /workspace/data/dev_phn
./utils/validate_data_dir.sh --no-feats /workspace/data/eval1_phn
```

## 3. Setting Up VITS with Speaker Embeddings

### 3.1. Enabling Speaker Embeddings and IDs

For multi-speaker training, enable speaker embeddings and assign unique speaker IDs (SIDs). If using x-vectors from Kaldi:

```bash
./run.sh --stage 2 --stop-stage 3 --use_spk_embed true --spk_embed_tool kaldi --use_sid true --nj $(nproc)
```

Explanation of flags:

- `--use_spk_embed true`: Enables extraction of speaker embeddings to differentiate voices in multi-speaker datasets.
- `--spk_embed_tool kaldi`: Specifies Kaldi as the tool for x-vector embedding extraction, utilizing pre-trained x-vector embeddings.
- `--use_sid true`: Assigns a unique Speaker ID (SID) to each speaker, helping the model distinguish between them.

## 4. Training the VITS Model

### 4.1 Data Prepartion

Before training, run the steps needed to generate statistics and other prerequisites:

```bash
nohup ./run.sh --stage 4 --stop-stage 6 \
    --tag vits_xvector_maltese \
    --nj $(nproc) \
    --ngpu <gpus_count> > nohup.out 2>&1 &
```

Usually, once you have run these steps and generated `exp/tts_stats_raw_phn_maltese_none` and `exp/xvector_nnet_1a`, you can reuse them for multiple experiments without re-running earlier stages. The exception is if you change data representation parameters (e.g., Mel filters, FFT size, hop length, or phoneme tokenization), in which case you must re-run from the appropriate earlier stage to regenerate consistent data.

### 4.2. Train Models Using A Configuration

To find an optimal configuration that leads to stable diagonal attention alignments and improved model generalization, you may need to train multiple times using different YAML configuration files. Each configuration can adjust training parameters—such as dropout rates, learning rates, or model architecture details—to refine the model’s performance.

Because the data preparation steps (stages 1–6) have already been completed, you can start each new experimental run directly from stage 7. This allows you to reuse the previously generated features and speaker embeddings without repeating the entire data processing pipeline. Simply modify the paths and parameters below as needed:

```bash
nohup ./run.sh --stage 7 --stop-stage 7 \
    --use_spk_embed true --spk_embed_tool kaldi --use_sid true \
    --train_config path/to/train/config.yaml \
    --tag <experiment tag> \
    --nj $(nproc) \
    --ngpu <gpus_count> > nohup.out 2>&1 &
```

- `--train_config` specifies the configuration file you wish to experiment with.
- `--tag` allows you to label each run distinctly.
- `--nj` sets the number of CPU jobs and
- `--ngpu` sets the number of GPUs to use.

After running this command, monitor the results, including the attention plots, to determine whether the configuration yields the desired diagonal alignment and overall stability. If not, adjust your configuration and rerun the training until you achieve the optimal setup.

#### Log Monitoring:

```bash
tail -f exp/<experiment tag>/train.log
```

## 5. Inference and Evaluation

After training, run inference to generate synthetic speech and evaluate results:

```bash
./run.sh --stage 7 --stop-stage 8 --inference-config conf/decode.yaml --inference_tag test_with_vits
```

Use the generated outputs to assess audio quality and attention alignment, refining configurations as necessary in subsequent training runs.

**In summary:**

- You set up the environment and dependencies once.
- Data preparation and feature extraction steps (stages 1–6) need only be re-run if you change how data is represented.
- Model training (stage 7) can be repeated with different configurations to improve model quality.
- Inference (stage 8) allows you to validate model performance and guide further refinements.

This process streamlines experimentation, letting you quickly iterate toward a stable, high-quality VITS-based TTS model.
