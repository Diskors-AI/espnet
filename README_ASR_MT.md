# ASR Training Workflow

This guide provides step-by-step instructions for setting up and training an Automatic Speech Recognition (ASR) model using ESPnet, with a focus on the VCTK/ASR1 recipe. You will replace the work typically done in Stage 1 with your own dataset.

## 1. Initial Setup

### 1.1. Directory Structure

Ensure the following directory structure in your workspace:

```
/workspace/
├── espnet/             # ESPnet repository
├── exp/                # Experiment results and trained models
├── dump/               # Processed data (features, etc.)
└── data/               # Raw data (audio + transcription)
    └── corpus/         # Corpus data (WAV files + alignments)
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

### 1.5. Install Python Requirements

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
pip install -e ".[asr]" # Install ASR dependencies
pip install kaldiio torchaudio praatio sox tensorboard matplotlib
```

### 1.6. Install Kaldi

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

### 1.7. Symlink Directories

Link the data, dump, and exp directories in the ESPnet setup:

```bash
ln -s /workspace/data /workspace/espnet/egs2/vctk/asr1/data
ln -s /workspace/dump /workspace/espnet/egs2/vctk/asr1/dump
ln -s /workspace/exp /workspace/espnet/egs2/vctk/asr1/exp
```

### 2.2. Data Preparation from TextGrid (MFA Alignments)

You should have generated your own dataset that replaces the functionality of Stage 1 from the VCTK/ASR1 recipe. To proceed with your data, structure your data into Kaldi-style directories:

```bash
/workspace/data/
├── train/
│   ├── wav.scp       # List of WAV files
│   ├── text          # Transcriptions
│   ├── utt2spk       # Utterance-to-speaker mapping
│   └── spk2utt       # Speaker-to-utterance mapping
├── dev/
├── eval/
├── eval_noisy/
└── eval_low_quality/

```

### 2.3. Validate Data

Validate that your prepared data is correct using ESPnet scripts:

```bash
cd /workspace/espnet/egs2/vctk/asr1
./utils/validate_data_dir.sh --no-feats data/train
./utils/validate_data_dir.sh --no-feats data/dev
./utils/validate_data_dir.sh --no-feats data/eval
./utils/validate_data_dir.sh --no-feats data/eval_noisy
./utils/validate_data_dir.sh --no-feats data/eval_low_quality
```

## 3. Feature Extraction

After data preparation, run the feature extraction for your ASR model:

```bash
./run.sh --stage 2 --stop-stage 2
```

This step will extract features like MFCC from your audio files.

## 4. Language Model Preparation

If you plan to use a language model, train it or use a pre-trained one. You can skip this step if you don't need a custom LM.

To train a language model, execute:

```bash
./run.sh --stage 3 --stop-stage 3
```

## 5. Training the ASR Model

Now you're ready to start training your ASR model:

```bash
./run.sh --stage 4 --stop-stage 6 --ngpu <num_of_gpus>
```

- `--stage 4`: Initiates the training process.
- `--stop-stage 6`: Stops after training and validation.
- `--ngpu`: Set the number of GPUs available for training.

Monitor your logs to ensure the model is training properly.

## 6. Decoding and Evaluation

Once training is complete, you can decode and evaluate the ASR model:

```bash
./run.sh --stage 7 --stop-stage 8
```

Check the output in the `exp` directory for results.

## 7. Optional Fine-tuning or Transfer Learning

If you have additional data or want to fine-tune your model, you can run additional training stages, starting with your trained model weights.

```bash
./run.sh --stage 4 --resume <path_to_checkpoint>
```

## 8. Inference

Run inference using your trained ASR model:

```bash
./run.sh --stage 8 --inference-config conf/decode.yaml
```

This will output transcriptions for your test dataset.

## 9. Logging and Monitoring

For real-time monitoring of training, you can use TensorBoard:

```bash
tensorboard --logdir exp/
```

---

This guide sets up ASR training using your custom dataset. If any issues arise, check the logs and the ESPnet documentation for troubleshooting help.
