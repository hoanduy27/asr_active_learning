python -m asr_evaluate.hypothesis.deepspeech2_hypothesizer \
    --ip 127.0.0.1 \
    --port 10000 \
    --model_code deepspeech2 \
    --data_path dummy_aal \
    --wav_meta_path dummy_aal/wav.csv \
    --report_dir dummy_aal/reports