import logging
import os
import glob
import numpy as np
import torch
import wave

from yacs import config as CONFIG
from transformers import AutoTokenizer

# Make sure these imports match your local paths.
from frontend import g2p_cn_en, ROOT_DIR, read_lexicon, G2p
from config.joint.config import Config
from models.prompt_tts_modified.jets import JETSGenerator
from models.prompt_tts_modified.simbert import StyleEncoder

logging.basicConfig(level=logging.INFO)

logger = logging.getLogger(__name__)

# See if metal is available, otherwise use CPU

if torch.backends.mps.is_available():
    device = torch.device("mps")
    logger.info("Using MPS")
else:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using {device}")
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
    style_encoder_checkpoint_path = scan_checkpoint(
        f'{config.output_directory}/style_encoder/ckpt',
        'checkpoint_',
        6
    )

    with open(config.model_config_path, 'r') as fin:
        conf = CONFIG.load_cfg(fin)
    conf.n_vocab = config.n_symbols
    conf.n_speaker = config.speaker_n_labels

    # Initialize models
    style_encoder = StyleEncoder(config)
    model_ckpt = torch.load(style_encoder_checkpoint_path, map_location="cpu", weights_only=True)
    style_encoder_dict = {}
    for key, value in model_ckpt['model'].items():
        style_encoder_dict[key[7:]] = value  # remove "module." prefix
    style_encoder.load_state_dict(style_encoder_dict, strict=False)
    style_encoder.eval()

    generator = JETSGenerator(conf).to(device)
    gen_ckpt = torch.load(am_checkpoint_path, map_location=device, weights_only=True)
    generator.load_state_dict(gen_ckpt['generator'])
    generator.eval()

    tokenizer = AutoTokenizer.from_pretrained(config.bert_path)

    with open(config.token_list_path, 'r') as f:
        lines = f.readlines()
        token2id = {t.strip(): idx for idx, t in enumerate(lines)}

    with open(config.speaker2id_path, encoding='utf-8') as f:
        lines = f.readlines()
        speaker2id = {t.strip(): idx for idx, t in enumerate(lines)}

    return style_encoder, generator, tokenizer, token2id, speaker2id

def get_style_embedding(prompt, tokenizer, style_encoder):
    """Embeds a text prompt using the style encoder."""
    prompt_inputs = tokenizer([prompt], return_tensors="pt")
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

    # If you want prompt vs content as separate embeddings, you can do both:
    style_embedding = get_style_embedding(prompt, tokenizer, style_encoder)
    content_embedding = get_style_embedding(text, tokenizer, style_encoder)

    # Convert speaker ID
    spkid = speaker2id[speaker]

    # G2P to produce token IDs:
    lexicon = read_lexicon(f"{ROOT_DIR}/lexicon/librispeech-lexicon.txt")
    g2p = G2p()
    text_tokens = g2p_cn_en(text, g2p, lexicon)

    text_int = [token2id[ph] for ph in text_tokens.split()]
    sequence = torch.from_numpy(np.array(text_int)).long().unsqueeze(0).to(device)
    sequence_len = torch.tensor([len(text_int)]).to(device)

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

def main():
    # Load config
    config = Config()

    # Load models
    models = get_models(config)

    # The text you want to synthesize
    text = "This is so exciting!"
    tone_mapping = {
        "Affectionate": {"Chinese": "充满爱意的", "Pinyin": "chōngmǎn àiyì de"},
        "Angry": {"Chinese": "生气的", "Pinyin": "shēngqì de"},
        "Annoyed": {"Chinese": "恼怒的", "Pinyin": "nǎonù de"},
        "Anxious": {"Chinese": "焦虑的", "Pinyin": "jiāolǜ de"},
        "Bored": {"Chinese": "无聊的", "Pinyin": "wúliáo de"},
        "Calm": {"Chinese": "平静的", "Pinyin": "píngjìng de"},
        "Cheerful": {"Chinese": "愉快的", "Pinyin": "yúkuài de"},
        "Confident": {"Chinese": "自信的", "Pinyin": "zìxìn de"},
        "Confused": {"Chinese": "困惑的", "Pinyin": "kùnhuò de"},
        "Cynical": {"Chinese": "愤世嫉俗的", "Pinyin": "fèn shì jì sú de"},
        "Defensive": {"Chinese": "防御的", "Pinyin": "fángyù de"},
        "Desperate": {"Chinese": "绝望的", "Pinyin": "juéwàng de"},
        "Determined": {"Chinese": "坚定的", "Pinyin": "jiāndìng de"},
        "Disappointed": {"Chinese": "失望的", "Pinyin": "shīwàng de"},
        "Disgusted": {"Chinese": "厌恶的", "Pinyin": "yànwù de"},
        "Enthusiastic": {"Chinese": "热情的", "Pinyin": "rèqíng de"},
        "Excited": {"Chinese": "兴奋的", "Pinyin": "xīngfèn de"},
        "Fearful": {"Chinese": "害怕的", "Pinyin": "hàipà de"},
        "Friendly": {"Chinese": "友好的", "Pinyin": "yǒuhǎo de"},
        "Frustrated": {"Chinese": "沮丧的", "Pinyin": "jǔsàng de"},
        "Grateful": {"Chinese": "感激的", "Pinyin": "gǎnjī de"},
        "Hopeful": {"Chinese": "充满希望的", "Pinyin": "chōngmǎn xīwàng de"},
        "Humorous": {"Chinese": "幽默的", "Pinyin": "yōumò de"},
        "Impatient": {"Chinese": "不耐烦的", "Pinyin": "bù nàifán de"},
        "Indifferent": {"Chinese": "冷漠的", "Pinyin": "lěngmò de"},
        "Inquisitive": {"Chinese": "好奇的", "Pinyin": "hàoqí de"},
        "Inspirational": {"Chinese": "鼓舞人心的", "Pinyin": "gǔwǔ rénxīn de"},
        "Jealous": {"Chinese": "嫉妒的", "Pinyin": "jídù de"},
        "Joyful": {"Chinese": "欢乐的", "Pinyin": "huānlè de"},
        "Melancholic": {"Chinese": "忧郁的", "Pinyin": "yōuyù de"},
        "Mocking": {"Chinese": "嘲讽的", "Pinyin": "cháofěng de"},
        "Motivational": {"Chinese": "激励的", "Pinyin": "jīlì de"},
        "Neutral": {"Chinese": "中性的", "Pinyin": "zhōngxìng de"},
        "Nostalgic": {"Chinese": "怀旧的", "Pinyin": "huáijiù de"},
        "Optimistic": {"Chinese": "乐观的", "Pinyin": "lèguān de"},
        "Pessimistic": {"Chinese": "悲观的", "Pinyin": "bēiguān de"},
        "Playful": {"Chinese": "爱玩的", "Pinyin": "àiwán de"},
        "Sarcastic": {"Chinese": "讽刺的", "Pinyin": "fěngcì de"},
        "Serious": {"Chinese": "严肃的", "Pinyin": "yánsù de"},
        "Skeptical": {"Chinese": "怀疑的", "Pinyin": "huáiyí de"},
        "Somber": {"Chinese": "阴郁的", "Pinyin": "yīnyù de"},
        "Sympathetic": {"Chinese": "同情的", "Pinyin": "tóngqíng de"},
        "Tender": {"Chinese": "温柔的", "Pinyin": "wēnróu de"},
        "Teasing": {"Chinese": "取笑的", "Pinyin": "qǔxiào de"},
        "Tense": {"Chinese": "紧张的", "Pinyin": "jǐnzhāng de"},
        "Thoughtful": {"Chinese": "体贴的", "Pinyin": "tǐtiē de"},
        "Urgent": {"Chinese": "紧急的", "Pinyin": "jǐnjí de"},
        "Warm": {"Chinese": "温暖的", "Pinyin": "wēnnuǎn de"},
        "Whimsical": {"Chinese": "异想天开的", "Pinyin": "yìxiǎngtiānkāi de"},
        "Worried": {"Chinese": "担心的", "Pinyin": "dānxīn de"}
    }
    prompt = tone_mapping['Cheerful']['Chinese']
    # Replace this with a valid speaker name from config.speakers
    speaker_name = config.speakers[0]  # or "English_speaker" if that exists

    # Generate the waveform
    audio_data = tts(text, prompt, speaker_name, models, config)

    # Write the waveform to a .wav file
    output_file = "tts_output.wav"
    with wave.open(output_file, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)  # 16-bit
        wf.setframerate(config.sampling_rate)
        wf.writeframes(audio_data.tobytes())

    print(f"Generated TTS audio saved to {output_file}")

if __name__ == "__main__":
    main()
