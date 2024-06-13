python -m asr_evaluate.hypothesis.wav2vec2_hypothesizer \
    --model_path nguyenvulebinh/wav2vec2-base-vietnamese-250h \
    --model_code wav2vec2 \
    --data_path dummy_aal \
    --wav_meta_path dummy_aal/wav.csv \
    --report_dir dummy_aal/reports