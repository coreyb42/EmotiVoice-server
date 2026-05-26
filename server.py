#!/usr/bin/env python3

import os
import io
import wave
import logging
import re
from flask import Flask, request, send_file, jsonify
from werkzeug.exceptions import BadRequest

# Adjust these imports to match your project structure
from config.joint.config import Config
from text.cleaners import english_cleaners
from tones import tone_mapping
from tts import get_models, tts

# Ollama for LLM text transformation
from ollama import chat, ChatResponse

# Initialize Flask app
app = Flask(__name__)

# Configure logging
logging.basicConfig(level=logging.INFO)

# Load configurations and models at startup
try:
    config = Config()
    style_encoder, generator, tokenizer, token2id, speaker2id = get_models(config)
    models = (style_encoder, generator, tokenizer, token2id, speaker2id)
    logging.info("TTS models loaded successfully.")
except Exception as e:
    logging.error(f"Failed to load configurations or models: {e}")
    raise

PAUSE_DURATION_SECONDS = 0.3  # Duration of pause between segments in seconds

# Define the LLM prompt template
LLM_PROMPT_TEMPLATE = """You are an assistant specialized in preparing text for Text-to-Speech (TTS) systems. 
Your task is to transform the provided input text into a version that is clear and natural for spoken delivery. 

**Do not explain - only return one copy of the transformed text, and nothing else. If there are no transformations to do, return a single copy of the text as-is.**

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

8. **Avoid explaining**: Your output should be clean and will not include any explanations or markup - we appreciate your work, and trust your expertise.

**Input Text:**

"{text}"
"""


def initialize_speaker(speaker_input):
    """
    Determine the speaker name and ID based on the input.
    """
    speaker_name = None
    speaker_id = None
    # Attempt to interpret speaker_input as ID
    try:
        speaker_id = int(speaker_input)
        speaker_name = next((name for name, sid in speaker2id.items() if sid == speaker_id), None)
    except ValueError:
        # Not an integer, assume it's a name
        if speaker_input in speaker2id:
            speaker_name = speaker_input
            speaker_id = speaker2id[speaker_name]
    if speaker_name is None:
        raise ValueError(f"Cannot find a speaker for '{speaker_input}' in speaker2id. "
                         "Check your speaker name or ID.")
    return speaker_name, speaker_id


def parse_text_with_tones(text):
    """
    Parses the input text for tone tags and returns a list of (tone, text) tuples.
    Supports only one level of tags (no nested tags).

    Example:
    Input: "<Happy>It's great!</Happy><Confused>What's going on?</Confused>"
    Output: [("Happy", "It's great!"), ("Confused", "What's going on?")]
    """
    pattern = re.compile(r'<(\w+)>(.*?)<\/\1>', re.DOTALL)
    matches = pattern.findall(text)
    segments = []
    last_end = 0

    for match in pattern.finditer(text):
        tone, segment_text = match.group(1), match.group(2)
        start, end = match.span()
        # Text before the current tag
        if start > last_end:
            pre_text = text[last_end:start].strip()
            if pre_text:
                segments.append((None, pre_text))  # None indicates default tone
        segments.append((tone, segment_text.strip()))
        last_end = end

    # Text after the last tag
    if last_end < len(text):
        post_text = text[last_end:].strip()
        if post_text:
            segments.append((None, post_text))  # None indicates default tone

    return segments


def generate_silence(duration_seconds, sampling_rate, channels, sample_width):
    """
    Generates silence audio bytes for the specified duration.

    :param duration_seconds: Duration of silence in seconds.
    :param sampling_rate: Sample rate of the audio (e.g., 22050).
    :param channels: Number of audio channels (1 for mono, 2 for stereo).
    :param sample_width: Sample width in bytes (2 for 16-bit audio).
    :return: Bytes representing silence audio.
    """
    num_samples = int(sampling_rate * duration_seconds)
    silence = b'\x00' * num_samples * channels * sample_width
    return silence


def generate_audio_for_segment(speaker_name, speaker_id, tone, segment_text, models, config):
    """
    Generates TTS audio for a single text segment with the specified tone.
    Returns the audio data as bytes.
    """
    # Validate the tone key
    if tone and tone not in tone_mapping:
        raise ValueError(f"Tone '{tone}' not found in tone_mapping.")

    # Prepare the style prompt from tone_mapping
    if tone:
        style_prompt = tone_mapping[tone].get("chinese", "")
    else:
        # Use a default tone if no tone is specified
        default_tone = "neutral"
        style_prompt = tone_mapping[default_tone].get("chinese", "")
        tone = default_tone  # Assign default tone name

    # Prepare the LLM prompt
    llm_input = LLM_PROMPT_TEMPLATE.format(text=segment_text)

    # Run the text through Ollama
    try:
        llm_response: ChatResponse = chat(
            model='llama3.2',
            messages=[{"role": "user", "content": llm_input}],
        )
        processed_text = llm_response.message.content.strip()
        logging.info(f"Processed text: {processed_text}")
    except Exception as e:
        logging.error(f"LLM processing failed: {e}")
        raise RuntimeError("Failed to process text with LLM.")

    # Optionally clean the processed text
    try:
        processed_text = english_cleaners(processed_text)
    except Exception as e:
        logging.warning(f"Text cleaning failed: {e}")
        # Proceed without cleaning if it fails
        processed_text = processed_text  # No change

    # Generate TTS audio
    logging.info(f"Generating TTS for speaker '{speaker_name}' (ID={speaker_id}), tone '{tone}'.")
    try:
        audio_data = tts(processed_text, style_prompt, speaker_name, models, config)
    except Exception as e:
        logging.error(f"TTS generation failed: {e}")
        raise RuntimeError("Failed to generate TTS audio.")

    return audio_data.tobytes()


@app.route('/generate_tts', methods=['POST'])
def generate_tts():
    """
    Endpoint to generate TTS audio from provided text.
    Supports specifying tones within the text using XML-like tags.
    Inserts small pauses between segments with different tones.
    """
    try:
        data = request.get_json()
        if not data:
            raise BadRequest("Invalid JSON payload.")

        speaker_input = data.get('speaker')
        tone = data.get('tone')  # Default tone
        text = data.get('text')

        # Validate input parameters
        if not speaker_input:
            return jsonify({"error": "Missing 'speaker' parameter."}), 400
        if not text:
            return jsonify({"error": "Missing 'text' parameter."}), 400

        logging.info(f"Received request - Speaker: {speaker_input}, Tone: {tone}")

        # Initialize speaker
        try:
            speaker_name, speaker_id = initialize_speaker(speaker_input)
        except ValueError as ve:
            logging.error(str(ve))
            return jsonify({"error": str(ve)}), 400

        # Determine if the text contains tone tags
        segments = parse_text_with_tones(text)
        has_tone_tags = any(tone_tag for tone_tag, _ in segments)

        if has_tone_tags:
            logging.info("Detected tone tags in the input text.")
            # If tone tags are present, the 'tone' parameter is ignored
            audio_segments = []
            for idx, (segment_tone, segment_text) in enumerate(segments):
                try:
                    audio_bytes = generate_audio_for_segment(
                        speaker_name, speaker_id, segment_tone, segment_text, models, config
                    )
                    audio_segments.append(audio_bytes)

                    # Insert pause after each segment except the last one
                    if idx < len(segments) - 1:
                        pause = generate_silence(
                            PAUSE_DURATION_SECONDS,
                            config.sampling_rate,
                            1,  # Mono
                            2   # 16-bit audio
                        )
                        audio_segments.append(pause)
                except Exception as e:
                    logging.error(f"Error processing segment: {e}")
                    return jsonify({"error": str(e)}), 500

            # Concatenate all audio segments with pauses
            concatenated_audio = b''.join(audio_segments)

            # Write concatenated audio to an in-memory buffer
            try:
                wav_buffer = io.BytesIO()
                with wave.open(wav_buffer, "wb") as wf:
                    # Assuming all segments have the same audio parameters
                    wf.setnchannels(1)  # Mono
                    wf.setsampwidth(2)  # 16-bit audio
                    wf.setframerate(config.sampling_rate)
                    wf.writeframes(concatenated_audio)
                wav_buffer.seek(0)
            except Exception as e:
                logging.error(f"Failed to write concatenated WAV data: {e}")
                return jsonify({"error": "Failed to process audio data."}), 500

            # Generate a filename based on speaker and tones
            filename = f"tts_output_speaker{speaker_id}_multiple_tones.wav"

            logging.info(f"Successfully generated concatenated TTS audio: {filename}")

            # Return the concatenated WAV file as a response
            return send_file(
                wav_buffer,
                as_attachment=True,
                download_name=filename,
                mimetype="audio/wav"
            )
        else:
            logging.info("No tone tags detected in the input text. Processing as single tone.")
            # No tone tags; process as a single segment
            if not tone:
                # Use default tone if not provided
                default_tone = "neutral"
                style_prompt = tone_mapping[default_tone].get("chinese", "")
                tone = default_tone
                logging.info(f"No tone specified. Using default tone: '{default_tone}'.")
            else:
                # Validate the tone key
                if tone not in tone_mapping:
                    error_msg = f"Tone '{tone}' not found in tone_mapping."
                    logging.error(error_msg)
                    return jsonify({"error": error_msg}), 400
                style_prompt = tone_mapping[tone].get("chinese", "")

            # Prepare the LLM prompt
            llm_input = LLM_PROMPT_TEMPLATE.format(text=text)

            # Run the text through Ollama
            try:
                llm_response: ChatResponse = chat(
                    model='phi4',
                    messages=[{"role": "user", "content": llm_input}],
                )
                processed_text = llm_response.message.content.strip()
                logging.info(f"Processed text: {processed_text}")
            except Exception as e:
                logging.error(f"LLM processing failed: {e}")
                return jsonify({"error": "Failed to process text with LLM."}), 500

            # Optionally clean the processed text
            try:
                processed_text = english_cleaners(processed_text)
            except Exception as e:
                logging.warning(f"Text cleaning failed: {e}")
                # Proceed without cleaning if it fails

            # Generate TTS audio
            logging.info(f"Generating TTS for speaker '{speaker_name}' (ID={speaker_id}), tone '{tone}'.")
            try:
                audio_data = tts(processed_text, style_prompt, speaker_name, models, config)
                audio_bytes = audio_data.tobytes()
            except Exception as e:
                logging.error(f"TTS generation failed: {e}")
                return jsonify({"error": "Failed to generate TTS audio."}), 500

            # Write audio data to an in-memory buffer
            try:
                wav_buffer = io.BytesIO()
                with wave.open(wav_buffer, "wb") as wf:
                    wf.setnchannels(1)  # Mono
                    wf.setsampwidth(2)  # 16-bit audio
                    wf.setframerate(config.sampling_rate)
                    wf.writeframes(audio_bytes)
                wav_buffer.seek(0)
            except Exception as e:
                logging.error(f"Failed to write WAV data: {e}")
                return jsonify({"error": "Failed to process audio data."}), 500

            # Generate a filename or use a default
            filename = f"tts_output_speaker{speaker_id}_{tone}.wav"

            logging.info(f"Successfully generated TTS audio: {filename}")

            # Return the WAV file as a response
            return send_file(
                wav_buffer,
                as_attachment=True,
                download_name=filename,
                mimetype="audio/wav"
            )

    except BadRequest as br:
        logging.error(f"Bad request: {br}")
        return jsonify({"error": str(br)}), 400
    except Exception as e:
        logging.error(f"Unexpected error: {e}")
        return jsonify({"error": "An unexpected error occurred."}), 500


# Optionally, define a health check endpoint
@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({"status": "Service is up and running."}), 200


if __name__ == '__main__':
    # Run the Flask app
    # For production, consider using a WSGI server like Gunicorn
    app.run(host='0.0.0.0', port=5001, debug=False)
