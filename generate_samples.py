import logging
import os
import random
import wave

# Make sure these imports match your local paths.
from config.joint.config import Config
from text.cleaners import english_cleaners
from tts import get_models, tts

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    # Load config
    config = Config()

    # Load models
    try:
        style_encoder, generator, tokenizer, token2id, speaker2id = get_models(config)
    except Exception as e:
        logger.error(f"Failed to load models: {e}")
        return

    models = (style_encoder, generator, tokenizer, token2id, speaker2id)

    # The base text you want to synthesize
    phrase_dict = {
        "hello-world": "Hello, world!",
        "quick-brown-fox": "The quick brown fox jumps over the lazy dog.",
        "seashells-seashore": "She sells seashells by the seashore.",
        "subtle-city-sounds": "Can you hear the subtle sounds of the city at night?",
        "owe-twenty-five-dollars": "I owe you twenty-five dollars.",
        "dr-smith-appointment": "Dr. Smith will see you now.",
        "turn-to-page-394": "Please turn to page 394 in your textbook.",
        "temperature-thirty-two-celsius": "The temperature today is expected to reach thirty-two degrees Celsius.",
        "email-example": "Email me at example_user123@example-domain.com.",
        "five-oclock-somewhere": "It's 5 o'clock somewhere.",
        "watch-game-7-30-pm": "Are you going to watch the game at 7:30 PM?",
        "eiffel-tower-height": "The Eiffel Tower stands at 324 meters tall.",
        "buy-fruits": "I bought apples, oranges, bananas, and grapes.",
        "read-terms-conditions": "Read the 'Terms & Conditions' before proceeding.",
        "covid19-impact": "COVID-19 pandemic has changed the world.",
        "visit-openai": "Visit us at www.openai.com for more information.",
        "meeting-scheduled-march15": "The meeting is scheduled for next Monday, March 15th.",
        "scored-98-6-percent": "She scored 98.6% on her exam.",
        "water-freezes-0c": "It's a well-known fact that water freezes at 0°C.",
        "exclaimed-wow-amazing": "He exclaimed, 'Wow! That's amazing!'",
        "rsvp-july20": "Please RSVP by July 20th.",
        "stock-price-increase": "The stock price increased by 12.5% yesterday.",
        "gif-or-jif": "Is it pronounced 'GIF' or 'JIF'?",
        "package-arrival-time": "The package will arrive between 9:00 AM and 5:00 PM.",
        "three-bedroom-house": "They live in a three-bedroom, two-bath house.",
        "masters-degree-cs": "She has a master's degree in computer science.",
        "example-url": "The URL is https://www.example.com/page?query=test.",
        "schedule-30-min-meeting": "I need to schedule a 30-minute meeting.",
        "100m-dash-9-58": "He won the 100-meter dash in 9.58 seconds.",
        "backup-data": "Please don't forget to back up your data.",
        "recipe-flour-salt": "The recipe calls for 2 cups of flour and 1.5 teaspoons of salt.",
        "movies-tonight": "They're going to the movies tonight.",
        "raining-cats-dogs": "It's raining cats and dogs outside.",
        "ceo-speech": "The CEO's speech was both inspiring and informative.",
        "received-4gpa": "She received a 4.0 GPA this semester.",
        "isbn-number": "The ISBN number for the book is 978-3-16-148410-0.",
        "pass-the-salt": "Can you pass the salt, please?",
        "concert-8pm-sharp": "The concert starts at 8 PM sharp.",
        "drives-tesla-s": "He drives a Tesla Model S.",
        "cost-discount": "The cost is $19.99 after a 20% discount.",
        "married-twenty-years": "They've been married for twenty years.",
        "flight-aa1234": "The flight number is AA1234 departing at 6:45 AM.",
        "state-of-art-facility": "It's a state-of-the-art facility.",
        "refer-section-5-2": "Please refer to section 5.2 of the manual.",
        "temperature-minus5": "The temperature dropped to -5 degrees last night.",
        "avid-reader-sf": "She’s an avid reader of science fiction novels.",
        "package-weight": "The package weighs 2.5 kilograms.",
        "well-respected-member": "He's a well-respected member of the community.",
        "favorite-sports": "Their favorite sports are basketball and soccer.",
        "update-antivirus": "Remember to update your antivirus software regularly.",

        # Video Game Phrases
        "player-health-low": "Your health is low!",
        "enemy-approaching": "An enemy is approaching!",
        "level-up": "Congratulations! You've leveled up!",
        "mission-accomplished": "Mission accomplished!",
        "collecting-coins": "You have collected 100 coins.",
        "new-quest-available": "A new quest is available.",
        "item-purchased": "You have purchased a health potion.",
        "save-game-prompt": "Do you want to save your game?",
        "loading-level": "Loading the next level...",
        "achievement-unlocked": "Achievement unlocked: First Blood!",
        "game-over": "Game over. Try again?",
        "inventory-full": "Your inventory is full.",
        "power-up-activated": "Power-up activated!",
        "connection-lost": "Connection lost. Attempting to reconnect.",
        "tutorial-completed": "Tutorial completed. Welcome to the game!",

        # Virtual Companion Phrases
        "good-morning": "Good morning! How can I assist you today?",
        "reminder-meeting": "Don't forget you have a meeting at 3 PM.",
        "motivational-quote": "Keep pushing forward, you're doing great!",
        "weather-update": "Today's weather is sunny with a high of 25 degrees.",
        "music-play-request": "Sure, playing your favorite playlist now.",
        "task-completed": "You've successfully completed your task.",
        "how-can-i-help": "How can I help you today?",
        "news-headlines": "Here are the top news headlines for today.",
        "appointment-confirmed": "Your appointment has been confirmed for tomorrow.",
        "happy-birthday": "Happy Birthday! Wishing you a wonderful day.",
        "traffic-update": "There's heavy traffic on your usual route.",
        "sleep-reminder": "It's getting late. Maybe it's time to wind down.",
        "recipe-suggestion": "How about trying a new pasta recipe tonight?",
        "book-recommendation": "I recommend reading 'The Alchemist' by Paulo Coelho.",
        "exercise-reminder": "Time for your daily workout! Let's get moving.",

        # Weird Phrases
        "whispering-shadows": "The shadows whispered secrets to the silver moon.",
        "melting-time": "Time melted like ice under the sun's invisible gaze.",
        "echoes-of-silence": "Echoes of silence danced in the labyrinth of dreams.",
        "floating-mirrors": "Floating mirrors reflected the song of unseen galaxies.",
        "whirling-flowers": "Whirling flowers sang lullabies to the wandering stars.",
        "crimson-clouds": "Crimson clouds painted melodies across the twilight sky.",
        "twisted-moonlight": "Twisted moonlight bent the reality of the waking world.",
        "serene-chaos": "In the serene chaos, the mind found its rhythm.",
        "whispering-oceans": "Whispering oceans carried messages to the unknown depths.",
        "glowing-shadows": "Glowing shadows danced on the edge of perception.",
        "silent-symphony": "A silent symphony played in the heart of the night.",
        "liquid-mirror": "The liquid mirror reflected the dreams of the cosmos.",
        "twilight-dreamscape": "Twilight dreamscape where reality fades into fantasy.",
        "phantom-breeze": "A phantom breeze carried the scent of forgotten memories.",
        "dancing-stars": "Dancing stars wove patterns of light in the velvet darkness.",

    }

    # Define tone mapping
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

    # Define base output directory
    base_output_dir = "outputs"

    # Ensure the base output directory exists
    os.makedirs(base_output_dir, exist_ok=True)

    skip_speakers = [0, 4, 10, 15]

    # Iterate over all speakers, except the skip_speakers
    for speaker_name in config.speakers:
        # Get speaker ID
        speaker_id = speaker2id.get(speaker_name)
        if speaker_id is None:
            logger.warning(f"Speaker '{speaker_name}' not found in speaker2id mapping. Skipping.")
            continue

        if speaker_id in skip_speakers:
            logger.warning(f"Skipping Speaker {speaker_id}: '{speaker_name}'")
            continue

        logger.info(f"Processing Speaker {speaker_id}: '{speaker_name}'")

        # Iterate over all tones
        for tone_name, tone_info in tone_mapping.items():
            logger.info(f"  Processing Tone: '{tone_name}'")

            # Get Chinese prompt for the current tone
            prompt = tone_info['Chinese']

            base_text_short = random.choice(list(phrase_dict.keys()))
            base_text = phrase_dict[base_text_short]
            text = english_cleaners(base_text)

            logger.info(f"    Generated Text: '{text}'")
            logger.debug(f"    Chinese Prompt: '{prompt}'")

            # Generate the waveform
            try:
                audio_data = tts(text, prompt, speaker_name, models, config)
            except Exception as e:
                logger.error(f"    Failed to generate audio for Speaker {speaker_id}, Tone '{tone_name}': {e}")
                continue

            # Define the output subfolder path
            speaker_folder = f"speaker-{speaker_id}"
            # Let's modify to include both the tone and the base_text_short
            base_filename = f"{tone_name.lower().replace(' ', '-')}-{base_text_short}"
            output_folder = os.path.join(base_output_dir, speaker_folder)

            os.makedirs(output_folder, exist_ok=True)

            # Define the output file path
            output_file = os.path.join(output_folder, f"{base_filename}.wav")

            # Write the waveform to a .wav file
            try:
                with wave.open(output_file, "wb") as wf:
                    wf.setnchannels(1)
                    wf.setsampwidth(2)  # 16-bit
                    wf.setframerate(config.sampling_rate)
                    wf.writeframes(audio_data.tobytes())
                logger.info(f"    Generated TTS audio saved to '{output_file}'")
            except Exception as e:
                logger.error(f"    Failed to write audio file '{output_file}': {e}")

if __name__ == "__main__":
    main()
