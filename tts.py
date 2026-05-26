# Adapted from the original code in the EmotiVoice repository, and thus should be under the same license, copied below.
# Copyright 2023, YOUDAO
#           2024, Du Jing(thuduj12@163.com)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import glob
import logging
import os

import numpy as np
import torch
from g2p_en import G2p
from transformers import AutoTokenizer
from yacs import config as CONFIG

from text.g2p_english import ROOT_DIR, read_lexicon, get_eng_phoneme
from models.prompt_tts_modified.jets import JETSGenerator
from models.prompt_tts_modified.simbert import StyleEncoder

logger = logging.getLogger(__name__)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MAX_WAV_VALUE = 32768.0


def scan_checkpoint(cp_dir, prefix, c=8):
    """Finds the last checkpoint in a directory matching a pattern."""
    pattern = os.path.join(cp_dir, prefix + '?'*c)
    cp_list = glob.glob(pattern)
    if len(cp_list) == 0:
        return None
    return sorted(cp_list)[-1]


def get_models(config):
    """Loads style encoder & JETS generator, along with tokenizer and speaker2id."""
    am_checkpoint_path = scan_checkpoint(
        f'{config.output_directory}/prompt_tts_open_source_joint/ckpt', 'g_'
    )
    if am_checkpoint_path is None:
        logger.error("No acoustic model checkpoint found.")
        raise FileNotFoundError("Acoustic model checkpoint not found.")

    style_encoder_checkpoint_path = scan_checkpoint(
        f'{config.output_directory}/style_encoder/ckpt',
        'checkpoint_',
        6
    )
    if style_encoder_checkpoint_path is None:
        logger.error("No style encoder checkpoint found.")
        raise FileNotFoundError("Style encoder checkpoint not found.")

    with open(config.model_config_path, 'r') as fin:
        conf = CONFIG.load_cfg(fin)
    conf.n_vocab = config.n_symbols
    conf.n_speaker = config.speaker_n_labels

    # Initialize Style Encoder
    style_encoder = StyleEncoder(config)
    model_ckpt = torch.load(style_encoder_checkpoint_path, map_location="cpu", weights_only=True)
    style_encoder_dict = {}
    for key, value in model_ckpt['model'].items():
        style_encoder_dict[key[7:]] = value  # remove "module." prefix
    style_encoder.load_state_dict(style_encoder_dict, strict=False)
    style_encoder.to(device)
    style_encoder.eval()
    logger.info("Loaded Style Encoder.")

    # Initialize JETS Generator
    generator = JETSGenerator(conf).to(device)
    gen_ckpt = torch.load(am_checkpoint_path, map_location=device, weights_only=True)
    generator.load_state_dict(gen_ckpt['generator'])
    generator.eval()
    logger.info("Loaded JETS Generator.")

    # Initialize Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(config.bert_path)

    # Load Token to ID mapping
    with open(config.token_list_path, 'r') as f:
        lines = f.readlines()
        token2id = {t.strip(): idx for idx, t in enumerate(lines)}

    # Load Speaker to ID mapping
    with open(config.speaker2id_path, encoding='utf-8') as f:
        lines = f.readlines()
        speaker2id = {t.strip(): idx for idx, t in enumerate(lines)}

    return style_encoder, generator, tokenizer, token2id, speaker2id


def get_style_embedding(prompt, tokenizer, style_encoder):
    """Embeds a text prompt using the style encoder."""
    prompt_inputs = tokenizer([prompt], return_tensors="pt").to(device)
    with torch.no_grad():
        output = style_encoder(
            input_ids=prompt_inputs["input_ids"],
            token_type_ids=prompt_inputs["token_type_ids"],
            attention_mask=prompt_inputs["attention_mask"],
        )
    style_embedding = output["pooled_output"].cpu().squeeze().numpy()
    return style_embedding


def tts(text, prompt, speaker, models, config):
    """Generates waveform for a given text + style prompt + speaker."""
    style_encoder, generator, tokenizer, token2id, speaker2id = models

    # Embed style and content
    style_embedding = get_style_embedding(prompt, tokenizer, style_encoder)
    content_embedding = get_style_embedding(text, tokenizer, style_encoder)

    # Convert speaker ID
    spkid = speaker2id.get(speaker, None)
    if spkid is None:
        logger.error(f"Speaker '{speaker}' not found in speaker2id mapping.")
        raise ValueError(f"Speaker '{speaker}' not found.")

    # G2P to produce token IDs:
    lexicon_path = os.path.join(ROOT_DIR, "lexicon", "librispeech-lexicon.txt")
    if not os.path.exists(lexicon_path):
        logger.error(f"Lexicon file not found at {lexicon_path}.")
        raise FileNotFoundError(f"Lexicon file not found at {lexicon_path}.")
    lexicon = read_lexicon(lexicon_path)
    g2p = G2p()
    text_tokens = get_eng_phoneme(text, g2p, lexicon)

    # Convert tokens to IDs
    try:
        text_int = [token2id[ph] for ph in text_tokens.split()]
    except KeyError as e:
        logger.error(f"Token '{e.args[0]}' not found in token2id mapping.")
        raise ValueError(f"Token '{e.args[0]}' not found.")
    sequence = torch.from_numpy(np.array(text_int)).long().unsqueeze(0).to(device)
    sequence_len = torch.tensor([len(text_int)]).to(device)

    # Prepare embeddings and speaker ID
    style_embedding_t = torch.from_numpy(style_embedding).unsqueeze(0).to(device)
    content_embedding_t = torch.from_numpy(content_embedding).unsqueeze(0).to(device)
    speaker_t = torch.tensor([spkid]).to(device)

    with torch.no_grad():
        infer_output = generator(
            inputs_ling=sequence,
            inputs_style_embedding=style_embedding_t,
            input_lengths=sequence_len,
            inputs_content_embedding=content_embedding_t,
            inputs_speaker=speaker_t,
            alpha=1.0
        )

    audio = infer_output["wav_predictions"].squeeze() * MAX_WAV_VALUE
    audio = audio.cpu().numpy().astype('int16')
    return audio
