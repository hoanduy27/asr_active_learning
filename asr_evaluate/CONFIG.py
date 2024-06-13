# Audio configuration
SAMPLE_RATE = 16000

BLANK = '<empty>'
# Dataset const
UTTERANCE_ID = 'utt_id'
TEXT = 'text'
PATH = 'path'
SPEAKER_ID = 'spkid'
NAME = 'name'
GENDER = 'gender'
AGE = 'age'
REGION = 'region'
CONDITION = 'condition'
DEVICE = 'device'
HYPOTHESIS = 'hypothesis'
INFERENCE_TIME = 'inference_time'
START = 'start_s'
END = 'end_s'

# DEVICE values
MAIKA = 'maika'
MICRO = 'micro'

# CONDITION values
CLEAN = 'clean'
NORMAL = 'normal'
NOISY = 'noisy'


TIME_FORMAT = '%b_%d_%Y'
ERR_FORMAT = '%.2f'
# AAL
AAL_CONST = {
    UTTERANCE_ID: 'utt_id',
    TEXT: 'text',
    PATH: 'path',
    SPEAKER_ID: 'spkid',
    NAME: 'name',
    GENDER: 'gender',
    AGE: 'age',
    REGION: 'region',
    CONDITION: 'condition',
    DEVICE: 'device',
    START: 'start_s',
    END: 'end_s'
}

# Backend
DEVICE_MAPPER = 'device_mapper'
CONDITION_MAPPER = 'condition_mapper'
BACKEND_CONST = {
    UTTERANCE_ID: 0,
    TEXT: 1,
    PATH: 2,
}
BACKEND_UTILS = {
    # Level from PATH to extract information
    SPEAKER_ID: 2,
    DEVICE: 3,
    # Mapping DEVICE
    DEVICE_MAPPER:{
        'audio_server': MAIKA,
        'audio_mic': MICRO,
    },
    # Mapping CONDITION
    CONDITION_MAPPER:{
        'mic': CLEAN,
        'mkt': NOISY,
        'qc': NOISY
    }
}

# VAIS
ENDPOINT = "core-grpc-wideband.memobot.io:443"
API_KEY = "58bff9a2-dcd5-11ec-bec9-0242ac12000a"

# KALDI
KALDI_CONST = {
    UTTERANCE_ID: 0,
    SPEAKER_ID: 1,
    PATH: 1,
    TEXT: 1
}
