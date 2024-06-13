# How to run Hypothesizer

Each Hypothesizer requires different arugments. However, there are some arguments that are shared among those Hypothesizers.
- `--model_code`: Identity of the model
- `--data_path`: Path to directory that contains all audio files
- `--wav_meta_path` [OPTIONAL]: Path to AAL format. If not specified, the hypothesizer will predict all audio files that are in `data_path`
- `--report_dir`: Report directory. Report filename will be "`data_path`-`model_code`-`configID`-`report_date`.csv".
- `--report_chkpt` [OPTIONAL]: Path to old report file to restore result

# Table of Contents
1. [Google ASR](#google_asr)
2. [Wav2Vec 2.0 (Fairseq)](#wav2vec2-fairseq)

## Google ASR
This Hypothesizer do not require `model_path`. However, you must specify authentication credentials. Please follow these step
1. Provide authentication credentials
```
$ export GOOGLE_APPLICATION_CREDENTIALS="/path/to/keyfile.json"
```

2. Run the hypothesizer
```
$ cd /path/to/asr_model_testing

$ python -m asr_evaluate.hypothesis.google_asr_hypothesizer \
    --model_code googleASR \
    --data_path /path/to/audio_directory \
    --wav_meta_path /path/to/wav.csv \
    --report_dir /path/to/report_directory \
    --report_chkpt /path/to/report_checkpoint.csv
```