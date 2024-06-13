# asr_model_testing
This repo supports:
- **Test ASR model**: ESPNET2's Conformer, HuggingFace's Wav2Vec, Vosk, Google ASR, VAIS ASR. 

    Including: Generate hypothesis, compute error rate (CER, WER, SER)
- **Convert between kaldi format and aal format**


## 1 Instructions
### 1.1 Data format
- Data format must follow the form as in `dummy_aal` (aal format), including:
    - `wav.csv`:
    - `speaker.csv`: 
    - `utt.csv`:

### 1.2 Test ASR model
- Step 1 (OPTIONAL): Format recipes to aal format (currently supports kaldi format only)
    
    ``
    $ python -m aal_asr_evaluate.dataio.exporter -from kaldi -to aal -src /path/to/src/folder -dst /path/to/dest/folder
    ``
- Step 2: Generate hypothesis

    ``$ python -m asr_evaluate.hypothesis.<script_to_run> PARAMS``

    `<script_to_run>` is described below: 

    |Model|Framework|`<script_to_run>`|
    |:----:|:---------:|:---:|
    |Conformer|ESPNET|`conformerT_esp2_hypothesizer`|
    |DeepSpeech2|OpenSeq2Seq|`deepspech2_hypothesizer`|
    |Google ASR|(API)|`google_asr_hypothesizer`|
    |Wav2Vec2|Hugging Face|`wav2vec2_hypothesizer`|

    Please take a look at examples in `run_examples`.

- Step 3: Compute CER, WER, SER
``$ python -m statistics.error_rate --hyp_path HYP_PATH [--wav_meta WAV_META_PATH] --report_dir REPORT_DIR``
    - When `text` column appears in `hyp_path`, it is OPTIONAL to specify `wav_meta`. In this case, groundtruth will be taken from `wav_meta` instead of `hyp_path`. If `text` column not found, `wav_meta` is REQUIRED.
    - `wav_meta` follows **aal format**.

### 1.3 Combine many dataset
I am working on updating `exporter.py` to make better interface for data format combination/conversion. Currently, this script only support conversion (partly). Combination is still in developing.

If you wish to combine many dataset, you can do as the following.
Suppose:
- `ds1`: kaldi format
- `ds2`: aal format
- Destination format: kaldi format 
 
```python
from asr_evaluate.dataio.dataset import WavDataset as wd
from asr_evaluate.dataio.export_format import Exporter as exp

ds1 = wd.from_kaldi_format('/path/to/ds1/wav.scp', '/path/to/ds1/utt2spk', 'path/to/ds1/text')

ds2 = wd.from_aal_format('/path/to/ds1/wav.csv')

ds_all = ds1 + ds2

exp.to_kaldi_format(ds_all, '/path/to/destination/dir')

# If you wish to pipe with `sox` command
# exp.to_kaldi_format(ds_all, '/path/to/destination/dir', pipe_sox=True)

```

## [For research] Active learning for Conformer
I wrote a bunch of testing code for extracting features for Conformer to be used in active learning. Code is trashy, though, I know that :), so I prepare initial bash script for you to quickly ping point which code is currently useful for the pipeline.

All you need to do is read ALL the code in `scripts/`. 

Steps:
- Copy `scripts/local/`, `utils` directories to [ESPNET2](https://github.com/espnet/espnet) recipe (for example: `egs2/<dataset_name>/asr1`)

```sh
cp -r scripts/local/* egs2/<dataset_name>/asr1
cp -r utils/* egs2/<dataset_name>/asr1
```

- Copy  directory
Copy either `scripts/extract_grad.sh` OR `scripts/extract_loss.sh` to the same recipe as before.
```sh
cp scripts/extract_grad.sh egs2/<dataset_name>/asr1/extract.sh
```

- For EACH ROUND (for example, round n)
    - Prepare the feat folder: 

        ```sh
        cd egs2/<dataset_name>/asr1
        mkdir feats/round_<n>
        ```

    - Prepare the dataset (in AAL format)

    - Run the extractor code
        ```sh
        ./extract.sh --stage 1 --stop_stage 3
        ```

- Fix all bugs and have fun!