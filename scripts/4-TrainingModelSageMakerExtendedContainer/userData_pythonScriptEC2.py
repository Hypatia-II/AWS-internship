import json
from TTS.utils.io import load_config

CONFIG = load_config('TTS/tts/configs/config.json')
CONFIG['datasets'][0]['path'] = './LJSpeech-1.1/'
CONFIG['audio']['stats_path'] = None
CONFIG['output_path'] = '../'
with open('config.json', 'w') as fp:
    json.dump(CONFIG, fp)
