#!/usr/bin/env python3

import argparse
import os
import wave
import logging

# Adjust these imports to match your project structure
from config.joint.config import Config
from text.cleaners import english_cleaners
from tones import tone_mapping
from tts import get_models, tts

# Ollama for LLM text transformation
from ollama import ChatResponse

"""
Usage:

python generate_tts.py \
  --speaker "36" \
  --tone "Calm" \
  --text "Please RSVP by 12:00 PM to confirm your attendance at the CEO's meeting regarding the quarterly ROI." \
  --output "outputs/abbreviations_acronyms.wav"
  
  
python generate_tts.py \
  --speaker "37" \
  --tone "Professional" \
  --text "The company's revenue increased by 12.5% to reach $3,450,000 in Q4 2024." \
  --output "outputs/complex_numbers_financial_figures.wav"


python generate_tts.py \
  --speaker "1344" \
  --tone "Professional" \
  --text "Let's schedule the meeting on 03/15/2025 at 5:30 PM PST." \
  --output "outputs/dates_times_formats.wav"

python generate_tts.py \
  --speaker "1393" \
  --tone "Bored" \
  --text "Visit us at https://www.example-site.com/about-us?ref=homepage or email support@example-site.com for more information." \
  --output "outputs/urls_email_addresses.wav"

python generate_tts.py \
  --speaker "1436" \
  --tone "Friendly" \
  --text "Join us for a fiesta on July 20th at the parque central. Don't forget to bring your sombrero!" \
  --output "outputs/mixed_languages_code_switching.wav"

This one has issues with "bass":
python generate_tts.py \
  --speaker "1436" \
  --tone "Confident" \
  --text "The bass player decided to record a new track in the lead." \
  --output "outputs/homographs_context_dependent.wav"

This one has problems with "api" and "OAuth":
python generate_tts.py \
  --speaker "1436" \
  --tone "Determined" \
  --text "The API endpoint requires OAuth 2.0 authentication using JWT tokens for secure data transmission." \
  --output "outputs/technical_terms_jargon.wav"

python generate_tts.py \
  --speaker "17" \
  --tone "Excited" \
  --text "Wait, you didn't say you were coming to the party!" \
  --output "outputs/punctuation_intonation.wav"


python generate_tts.py \
  --speaker "Speaker1" \
  --tone "Thoughtful" \
  --text "Although the weather was unfavorable, the team decided to proceed with the outdoor event, ensuring all safety measures were in place." \
  --output "outputs/complex_sentences_multiple_clauses.wav"
  
  
  # 1. Abbreviations and Acronyms
python generate_tts.py --speaker "Speaker1" --tone "Confused" --text "Please RSVP by 12:00 PM to confirm your attendance at the CEO's meeting regarding the quarterly ROI." --output "outputs/abbreviations_acronyms.wav"

# 2. Complex Numbers and Financial Figures
python generate_tts.py --speaker "Speaker1" --tone "Professional" --text "The company's revenue increased by 12.5% to reach $3,450,000 in Q4 2024." --output "outputs/complex_numbers_financial_figures.wav"

# 3. Dates and Times in Various Formats
python generate_tts.py --speaker "Speaker1" --tone "Neutral" --text "Let's schedule the meeting on 03/15/2025 at 5:30 PM PST." --output "outputs/dates_times_formats.wav"

# 4. URLs and Email Addresses
python generate_tts.py --speaker "Speaker1" --tone "Informative" --text "Visit us at https://www.example-site.com/about-us?ref=homepage or email support@example-site.com for more information." --output "outputs/urls_email_addresses.wav"

# 5. Mixed Languages and Code-Switching
python generate_tts.py --speaker "Speaker1" --tone "Friendly" --text "Join us for a fiesta on July 20th at the parque central. Don't forget to bring your sombrero!" --output "outputs/mixed_languages_code_switching.wav"

# 6. Homographs and Context-Dependent Pronunciations
python generate_tts.py --speaker "Speaker1" --tone "Clarifying" --text "The bass player decided to record a new track in the lead." --output "outputs/homographs_context_dependent.wav"

# 7. Technical Terms and Jargon
python generate_tts.py --speaker "Speaker1" --tone "Technical" --text "The API endpoint requires OAuth 2.0 authentication using JWT tokens for secure data transmission." --output "outputs/technical_terms_jargon.wav"

# 8. Punctuation and Intonation for Natural Flow
python generate_tts.py --speaker "Speaker1" --tone "Expressive" --text "Wait, you didn't say you were coming to the party!" --output "outputs/punctuation_intonation.wav"

# 9. Complex Sentences with Multiple Clauses
python generate_tts.py --speaker "Speaker1" --tone "Thoughtful" --text "Although the weather was unfavorable, the team decided to proceed with the outdoor event, ensuring all safety measures were in place." --output "outputs/complex_sentences_multiple_clauses.wav"

# 10. Numerical Ranges and Fractions
python generate_tts.py --speaker "Speaker1" --tone "Informative" --text "The temperature will range from -5°C to 15°C, with a chance of precipitation at 30%." --output "outputs/numerical_ranges_fractions.wav"

# 11. Emotive and Expressive Language
python generate_tts.py --speaker "Speaker1" --tone "Excited" --text "Wow! That's absolutely incredible—I'm speechless!" --output "outputs/emotive_expressive_language.wav"

# 12. Long Compound Words and Portmanteaus
python generate_tts.py --speaker "Speaker1" --tone "Enthusiastic" --text "The hyperloop technology promises ultra-fast transportation across metropolitan areas." --output "outputs/compound_words_portmanteaus.wav"

# 13. Currency and Measurement Conversions
python generate_tts.py --speaker "Speaker1" --tone "Neutral" --text "The package weighs 2.5 kilograms and costs €49.99 to ship internationally." --output "outputs/currency_measurement_conversions.wav"

# 14. Nested Parentheses and Brackets
python generate_tts.py --speaker "Speaker1" --tone "Informative" --text "The results (as shown in Table 3 [see Appendix B]) indicate a significant improvement." --output "outputs/nested_parentheses_brackets.wav"

# 15. Slang, Idioms, and Colloquialisms
python generate_tts.py --speaker "Speaker1" --tone "Casual" --text "He's pulling my leg—there's no way that's actually happening!" --output "outputs/slang_idioms_colloquialisms.wav"

# 16. Multilingual Content with Code-Switching
python generate_tts.py --speaker "Speaker1" --tone "Bilingual" --text "Please review the document and enviar tus comentarios antes del viernes." --output "outputs/multilingual_code_switching.wav"

# 17. Homonyms and Wordplay
python generate_tts.py --speaker "Speaker1" --tone "Playful" --text "The knight rode through the night without a sound." --output "outputs/homonyms_wordplay.wav"

# 18. Complex Formatting and Symbols
python generate_tts.py --speaker "Speaker1" --tone "Technical" --text "The formula E=mc² revolutionized physics, showcasing the relationship between energy and mass." --output "outputs/complex_formatting_symbols.wav"

# 19. Nested Quotes and Punctuation
python generate_tts.py --speaker "Speaker1" --tone "Narrative" --text "She said, \"I heard him shout, 'Watch out!' just before the collision.\"" --output "outputs/nested_quotes_punctuation.wav"

# 20. Technical URLs with Query Parameters
python generate_tts.py --speaker "Speaker1" --tone "Informative" --text "Access the dashboard at https://dashboard.example.com/user/settings?theme=dark&lang=en." --output "outputs/technical_urls_query_params.wav"

# 21. Scientific Notation and Units
python generate_tts.py --speaker "Speaker1" --tone "Academic" --text "The experiment yielded a result of 3.14 × 10⁻⁴ mol/L under standard conditions." --output "outputs/scientific_notation_units.wav"

# 22a. Emotional Tone Variation - Excited
python generate_tts.py --speaker "Speaker1" --tone "Excited" --text "I'm so excited to start this new adventure!" --output "outputs/emotional_tone_excited.wav"

# 22b. Emotional Tone Variation - Worried
python generate_tts.py --speaker "Speaker1" --tone "Worried" --text "I'm really worried about the upcoming exam." --output "outputs/emotional_tone_worried.wav"

# 23. Multiple Languages in One Sentence
python generate_tts.py --speaker "Speaker1" --tone "Bilingual" --text "Welcome to the event! Bienvenidos a todos." --output "outputs/multiple_languages_single_sentence.wav"

# 24. Legal and Formal Language
python generate_tts.py --speaker "Speaker1" --tone "Formal" --text "Hereby, I declare that the information provided is true and accurate to the best of my knowledge." --output "outputs/legal_formal_language.wav"

# 25. Complex Acronyms and Initialisms
python generate_tts.py --speaker "Speaker1" --tone "Technical" --text "The NASA mission will utilize AI-driven UAVs for autonomous data collection." --output "outputs/complex_acronyms_initialisms.wav"


"""

# Same tone_mapping you showed in your bulk script

# This is the LLM prompt template you shared.
# Wrap it in triple quotes and keep it as a Python string literal.
LLM_PROMPT_TEMPLATE = """You are an assistant specialized in preparing text for Text-to-Speech (TTS) systems. 
Your task is to transform the provided input text into a version that is clear and natural for spoken delivery. 

**Do not explain - only return the transformed text, and nothing else.**

You will follow these steps:

1. **Expand Abbreviations and Acronyms:** 
   - **Full Expansion:** Convert all abbreviations and acronyms that have a full-word equivalent (e.g., "Dr.") to their full forms (e.g., "Doctor").
   - **Phonetic Spelling:** For abbreviations and acronyms that are typically pronounced as individual letters (e.g., "RSVP", "GPA"), spell them out letter by letter (e.g., "R S V P", "G P A").

2. **Replace Symbols with Words:** Substitute common symbols with their corresponding words. For example:
   - "$" → "dollars"
   - "%" → "percent"
   - "&" → "and"
   - "+" → "plus"
   - "@" → "at"
   - "#" → "number" or "hashtag" (context-dependent)

3. **Convert Numbers to Spoken Form:** Change numerical digits to their spoken equivalents. For example:
   - "123" → "one hundred twenty-three"
   - "3.14" → "three point one four"
   - "2025" → "two thousand twenty-five"

4. **Handle Dates and Times:** Convert dates and times into a spoken-friendly format.
   - "01/19/2025" → "January nineteenth, two thousand twenty-five"
   - "5:30 PM" → "five thirty PM"

5. **Spell Out Email Addresses and URLs:** 
   - **Email Addresses:** Replace "@" with "at" and "." with "dot". For example, "dr.smith@example.com" → "dr dot smith at example dot com".
   - **URLs:** Depending on the context, you may choose to spell them out or read them as they are. For example, "www.example.com" → "double-u double-u double-u dot example dot com".

6. **Ensure Natural Flow:** Adjust punctuation and phrasing to enhance readability and naturalness when spoken. For example, replace commas with pauses if necessary or rephrase sentences for clarity.

7. **Maintain Proper Names and Specific Terms:** Keep proper nouns, technical terms, and brand names unchanged unless a spoken equivalent is more appropriate.

8. **Avoid explaining**: Your output should be clean and will not include any explanations - we appreciate your work, and trust your expertise.

**Input Text:**

"{text}"
"""


def main():
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser(description="Generate a single TTS sample with a given speaker, tone, and text.")
    parser.add_argument("--speaker", type=str, required=True,
                        help="Speaker name (as in config) or a known speaker ID in speaker2id.")
    parser.add_argument("--tone", type=str, required=True,
                        help="Tone key from the tone_mapping dict (e.g. 'Neutral', 'Angry').")
    parser.add_argument("--text", type=str, required=True,
                        help="The plain text you want to transform and speak.")
    parser.add_argument("--output", type=str, required=True,
                        help="Path to the output .wav file.")

    args = parser.parse_args()

    # Load config
    config = Config()

    # Load TTS models
    style_encoder, generator, tokenizer, token2id, speaker2id = get_models(config)
    models = (style_encoder, generator, tokenizer, token2id, speaker2id)

    # Try to interpret speaker as a name or ID
    speaker_name = None
    # if args.speaker in speaker2id:
    #     # It's a known speaker name
    #     speaker_name = args.speaker
    #     speaker_id = speaker2id[speaker_name]
    # else:
        # Possibly the user passed the ordering id?
    try:
        speaker_id = int(args.speaker)
        speaker_name = next((name for name, sid in speaker2id.items() if sid == speaker_id), None)
    except ValueError:
        pass  # Not an integer

    if speaker_name is None:
        raise ValueError(f"Cannot find a speaker for '{args.speaker}' in speaker2id. "
                         "Check your speaker name or ID.")

    # Validate the tone key
    if args.tone not in tone_mapping:
        raise ValueError(f"Tone '{args.tone}' not found in tone_mapping.")

    # Prepare the Chinese (or other) style prompt from tone_mapping
    style_prompt = tone_mapping[args.tone]["chinese"]

    # 1) Use Ollama to expand and transform the text
    from ollama import chat

    # Insert your `args.text` into the LLM prompt
    llm_input = LLM_PROMPT_TEMPLATE.format(text=args.text)

    # Run it through Ollama (make sure 'phi4' or whichever model you want is installed and available)
    llm_response: ChatResponse = chat(
        model='phi4',
        messages=[{"role": "user", "content": llm_input}],
    )

    # Extract the processed text from the LLM response
    processed_text = llm_response.message.content.strip()

    # Optional: You might still want to run your text through english_cleaners
    # if your TTS pipeline expects some standard formatting.
    processed_text = english_cleaners(processed_text)

    # 2) Generate TTS audio
    logging.info(f"Generating TTS for speaker '{speaker_name}' (ID={speaker_id}), tone '{args.tone}'.")
    logging.info(f"Text after LLM transformation: {processed_text}")
    try:
        audio_data = tts(processed_text, style_prompt, speaker_name, models, config)
    except Exception as e:
        raise RuntimeError(f"Failed to generate TTS audio: {e}")

    # 3) Save to .wav
    output_path = args.output
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    try:
        with wave.open(output_path, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)  # 16-bit audio
            wf.setframerate(config.sampling_rate)
            wf.writeframes(audio_data.tobytes())
    except Exception as e:
        raise IOError(f"Failed to write WAV file '{output_path}': {e}")

    logging.info(f"Saved TTS audio to: {output_path}")


if __name__ == "__main__":
    main()
