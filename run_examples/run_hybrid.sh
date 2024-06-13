python -m asr_evaluate.hypothesis.vosk_hypothesizer \
    --model_path models/aal-s3 \
    --model_code hybrid \
    --data_path dummy_aal \
    --wav_meta_path dummy_aal/wav.csv \
    --report_dir dummy_aal/reports