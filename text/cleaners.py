""" 
Enhanced Cleaners Module for Text Normalization
Integrates comprehensive text normalization for improved TTS pronunciation.

From: https://github.com/keithito/tacotron (original)
"""

import re
from unidecode import unidecode
from num2words import num2words

# Regular expression matching whitespace:
_whitespace_re = re.compile(r'\s+')

# List of (regular expression, replacement) pairs for abbreviations:
_abbreviations = [(re.compile(r'\b%s\.' % x[0], re.IGNORECASE), x[1]) for x in [
    ('mrs', 'misess'),
    ('mr', 'mister'),
    ('dr', 'doctor'),
    ('st', 'saint'),
    ('co', 'company'),
    ('jr', 'junior'),
    ('maj', 'major'),
    ('gen', 'general'),
    ('drs', 'doctors'),
    ('rev', 'reverend'),
    ('lt', 'lieutenant'),
    ('hon', 'honorable'),
    ('sgt', 'sergeant'),
    ('capt', 'captain'),
    ('esq', 'esquire'),
    ('ltd', 'limited'),
    ('col', 'colonel'),
    ('ft', 'fort'),
]]

def expand_abbreviations(text):
    """Expand abbreviations using the predefined list."""
    for regex, replacement in _abbreviations:
        text = re.sub(regex, replacement, text)
    return text

def expand_contractions(text):
    """Expand common English contractions."""
    contractions = {
        "it's": "it is",
        "don't": "do not",
        "can't": "cannot",
        "she's": "she is",
        "they're": "they are",
        "he's": "he is",
        "i've": "i have",
        "you're": "you are",
        "we're": "we are",
        "they've": "they have",
        "I've": "I have",
        "You're": "You are",
        "We're": "We are",
        "They're": "They are",
        "He'd": "He had",
        "She'd": "She had",
        "I'd": "I had",
        "Don't": "Do not",
        "Can't": "Cannot",
        "It's": "It is",
        # Add more contractions as needed
    }
    contractions_pattern = re.compile('({})'.format('|'.join(re.escape(k) for k in contractions.keys())),
                                      flags=re.IGNORECASE | re.DOTALL)

    def replace(match):
        matched_text = match.group(0)
        lower_matched = matched_text.lower()
        replacement = contractions.get(lower_matched, matched_text)
        # Preserve the case of the first character
        if matched_text[0].isupper():
            replacement = replacement[0].upper() + replacement[1:]
        return replacement

    return contractions_pattern.sub(replace, text)


import re
from num2words import num2words


def normalize_numbers(text):
    """
    Convert numbers (cardinal and ordinal), decimals, and negative numbers to words.
    Special handling for ISBNs and phone numbers:
      - ISBNs: Each digit is spelled out with 'dash' separating segments.
      - Phone Numbers: Each digit is spelled out without 'dash'.
    Ensures that numbers are fully spelled out without hyphens, except where specified.
    """

    # ----------------------------
    # Step 1: Replace ISBNs
    # ----------------------------
    # ISBN patterns (supports ISBN-10 and ISBN-13 with various separators)
    isbn_pattern = re.compile(
        r'\b(?:ISBN(?:-1[03])?:?\s*)?(97[89][- ]?)?\d{1,5}[- ]\d{1,7}[- ]\d{1,7}[- ]\d\b',
        re.IGNORECASE
    )

    def replace_isbn(match):
        isbn = match.group()
        # Remove any ISBN prefix
        isbn = re.sub(r'^ISBN(?:-1[03])?:?\s*', '', isbn, flags=re.IGNORECASE)
        # Normalize separators to hyphens
        isbn = isbn.replace(' ', '-')
        # Replace each digit with its word equivalent and hyphens with 'dash'
        result = []
        for char in isbn:
            if char.isdigit():
                word = num2words(int(char))
                result.append(word)
            elif char in ['-', '–', '—']:
                result.append('dash')
            else:
                # Keep other characters as-is (if any)
                result.append(char)
        return ' '.join(result)

    text = isbn_pattern.sub(replace_isbn, text)

    # ----------------------------
    # Step 2: Replace Phone Numbers
    # ----------------------------
    # Phone number patterns (supports formats like 123-456-7890, (123) 456-7890)
    phone_pattern = re.compile(
        r'\b(?:\(\d{3}\)\s?|\d{3}[- ]?)\d{3}[- ]?\d{4}\b'
    )

    def replace_phone(match):
        phone = match.group()
        # Remove all non-digit characters
        digits = re.sub(r'\D', '', phone)
        # Replace each digit with its word equivalent
        words = [num2words(int(d)) for d in digits]
        return ' '.join(words)

    text = phone_pattern.sub(replace_phone, text)

    # ----------------------------
    # Step 3: Normalize Remaining Numbers
    # ----------------------------
    # Handle ordinal numbers (e.g., 20th -> 'twentieth')
    ordinal_pattern = re.compile(r'\b(\d+)(st|nd|rd|th)\b', re.IGNORECASE)
    text = ordinal_pattern.sub(lambda m: num2words(int(m.group(1)), ordinal=True), text)

    # Handle negative numbers (e.g., -5 -> 'negative five')
    # Ensure the negative sign is not part of a larger hyphenated pattern
    negative_pattern = re.compile(r'(?<![-\d])-\s*(\d+(\.\d+)?)\b')
    text = negative_pattern.sub(lambda m: 'negative ' + num2words(float(m.group(1))), text)

    # Handle numbers with decimals (e.g., 12.5 -> 'twelve point five')
    decimal_pattern = re.compile(r'\b(\d+)\.(\d+)\b')
    text = decimal_pattern.sub(
        lambda m: num2words(int(m.group(1))) + ' point ' + ' '.join(num2words(int(d)) for d in m.group(2)),
        text
    )

    # Handle cardinal numbers (e.g., 48 -> 'forty-eight')
    # Ensure numbers are not part of hyphenated words
    cardinal_pattern = re.compile(r'(?<![-\w])\b\d+\b(?![-\w])')
    text = cardinal_pattern.sub(lambda m: num2words(int(m.group())), text)

    return text


def expand_currencies(text):
    """Convert currency expressions to words."""
    currency_pattern = re.compile(r'\$(\d+(?:\.\d+)?)')

    def replace_currency(match):
        amount = match.group(1)
        try:
            if '.' in amount:
                dollars, cents = amount.split('.')
                dollars_word = num2words(int(dollars)) + " dollars"
                cents_word = num2words(int(cents)) + " cents"
                return f"{dollars_word} and {cents_word}"
            else:
                dollars_word = num2words(int(amount)) + " dollars"
                return dollars_word
        except:
            return match.group()

    return currency_pattern.sub(replace_currency, text)

def expand_percentages(text):
    """Convert percentage expressions to words."""
    percentage_pattern = re.compile(r'(\d+(\.\d+)?)%')

    def replace_percentage(match):
        number = match.group(1)
        try:
            number_word = num2words(float(number)).replace('-', ' ')
            return f"{number_word} percent"
        except:
            return match.group()

    return percentage_pattern.sub(replace_percentage, text)

def expand_dates(text):
    """Convert date expressions to words (e.g., July 20th -> July twentieth)."""
    date_pattern = re.compile(r'\b(January|February|March|April|May|June|July|August|September|October|November|December)\s+(\d{1,2})(st|nd|rd|th)\b', re.IGNORECASE)

    def replace_date(match):
        month = match.group(1)
        day = int(match.group(2))
        day_word = num2words(day, ordinal=True)
        return f"{month} {day_word}"

    return date_pattern.sub(replace_date, text)

def expand_times(text):
    """Convert time expressions to words (e.g., 7:30 PM -> seven thirty PM)."""
    time_pattern = re.compile(r'\b(\d{1,2}):(\d{2})\s*(AM|PM|am|pm)\b')

    def replace_time(match):
        hour = int(match.group(1))
        minute = int(match.group(2))
        period = match.group(3).upper()
        if minute == 0:
            return f"{num2words(hour)} {period}"
        else:
            return f"{num2words(hour)} {num2words(minute)} {period}"

    return time_pattern.sub(replace_time, text)

def expand_temperatures(text):
    """Convert temperature expressions to words (e.g., 48°C -> forty eight degrees Celsius)."""
    temp_pattern = re.compile(r'\b(-?\d+)\s*°\s*([CFK])\b', re.IGNORECASE)

    def replace_temp(match):
        temp = int(match.group(1))
        scale = match.group(2).upper()
        if scale == 'C':
            scale_word = 'degrees Celsius'
        elif scale == 'F':
            scale_word = 'degrees Fahrenheit'
        elif scale == 'K':
            scale_word = 'kelvin'
        else:
            scale_word = scale.lower()
        temp_word = num2words(abs(temp))
        prefix = "negative " if temp < 0 else ""
        return f"{prefix}{temp_word} {scale_word}"

    return temp_pattern.sub(replace_temp, text)

def expand_emails(text):
    """Convert email addresses to words (e.g., example_user123@example-domain.com -> example user one two three at example dash domain dot com)."""
    email_pattern = re.compile(r'([\w\._-]+)@([\w\.-]+)\.([\w]+)')

    def replace_email(match):
        username = match.group(1).replace('_', ' underscore ').replace('.', ' dot ')
        domain = match.group(2).replace('.', ' dot ').replace('-', ' dash ')
        tld = match.group(3).replace('.', ' dot ')
        return f"{username} at {domain} dot {tld}"

    return email_pattern.sub(replace_email, text)

def expand_urls(text):
    """Convert URLs to words (e.g., www.example.com -> www dot example dot com)."""
    url_pattern = re.compile(
        r'\b((?:https?://)?(?:www\.)?[\w\-]+(?:\.[\w\-]+)+[/\w\-\._~:/?#[\]@!$&\'()*+,;=]*)\b',
        re.IGNORECASE)

    def replace_url(match):
        url = match.group(1)
        # Remove protocol if present
        url = re.sub(r'^https?://', '', url, flags=re.IGNORECASE)
        # Split by '/'
        parts = url.split('/')
        # Process each part
        processed_parts = []
        for part in parts:
            # Replace dots and hyphens
            part = part.replace('.', ' dot ').replace('-', ' dash ')
            processed_parts.append(part)
        return ' slash '.join(processed_parts)

    return url_pattern.sub(replace_url, text)

def expand_isbn(text):
    """Convert ISBN numbers to words (e.g., ISBN 978-3-16-148410-0 -> ISBN nine seven eight dash three dash one six dash one four eight four one zero dash zero)."""
    isbn_pattern = re.compile(r'\bISBN\s+([\d\-]+)\b', re.IGNORECASE)

    def replace_isbn(match):
        isbn_number = match.group(1).replace('-', ' dash ')
        # Spell out each digit
        isbn_number = ' '.join(num2words(int(char)) if char.isdigit() else char for char in isbn_number.split())
        return f"ISBN {isbn_number}"

    return isbn_pattern.sub(replace_isbn, text)

def expand_special_cases(text):
    """Handle special cases like COVID-19, URLs with queries, etc."""
    # Handle COVID-19 as "COVID nineteen"
    covid_pattern = re.compile(r'\bCOVID[-]?(\d+)\b', re.IGNORECASE)
    text = covid_pattern.sub(lambda m: f"COVID {num2words(int(m.group(1)))}", text)

    # Add more special case handlers here as needed

    return text

def split_hyphenated(text):
    """Split hyphenated words into separate words."""
    hyphen_pattern = re.compile(r'\b\w+-\w+\b')
    return hyphen_pattern.sub(lambda m: ' '.join(m.group().split('-')), text)

def remove_punctuation(text):
    """Remove or replace punctuation for smoother speech."""
    # Replace apostrophes in possessives with ' of '
    # text = re.sub(r"(\w+)'s", r"\1 of", text)
    # Remove remaining punctuation
    # text = re.sub(r'[.,!?]', '', text)
    return text

def enhance_abbreviations(text):
    """Add periods after letters in certain abbreviations."""
    applicable = [
        'gpa',
        'rsvp'
    ]
    for abbr in applicable:
        text = text.replace(abbr, '.'.join(abbr))
    return text

def normalize_whitespace(text):
    """Collapse multiple whitespaces into a single space."""
    return re.sub(_whitespace_re, ' ', text).strip()

def convert_to_ascii(text):
    """Convert text to ASCII using unidecode."""
    return unidecode(text)

def lowercase(text):
    """Convert text to lowercase."""
    return text.lower()

def basic_cleaners(text):
    '''Basic pipeline that lowercases and collapses whitespace without transliteration.'''
    text = lowercase(text)
    text = normalize_whitespace(text)
    return text

def transliteration_cleaners(text):
    '''Pipeline for non-English text that transliterates to ASCII.'''
    text = convert_to_ascii(text)
    text = lowercase(text)
    text = normalize_whitespace(text)
    return text

def english_cleaners(text):
    '''Pipeline for English text, including number and abbreviation expansion.'''
    text = lowercase(text)
    text = expand_temperatures(text)
    text = expand_contractions(text)
    text = expand_abbreviations(text)
    text = expand_dates(text)
    text = expand_times(text)
    text = expand_currencies(text)
    text = expand_percentages(text)
    text = expand_emails(text)
    text = expand_urls(text)
    text = expand_isbn(text)
    text = expand_special_cases(text)
    text = normalize_numbers(text)
    text = split_hyphenated(text)
    text = remove_punctuation(text)
    text = enhance_abbreviations(text)
    text = normalize_whitespace(text)
    text = convert_to_ascii(text)
    return text

# Example usage:
if __name__ == "__main__":
    phrases = [
        "Hello, world!",
        "The quick brown fox jumps over the lazy dog.",
        "She sells seashells by the seashore.",
        "Can you hear the subtle sounds of the city at night?",
        "I owe you twenty-five dollars.",
        "Dr. Smith will see you now.",
        "Please turn to page 394 in your textbook.",
        "The temperature today is expected to reach thirty-two degrees Celsius.",
        "Email me at example_user123@example-domain.com.",
        "It's 5 o'clock somewhere.",
        "Are you going to watch the game at 7:30 PM?",
        "The Eiffel Tower stands at 324 meters tall.",
        "I bought apples, oranges, bananas, and grapes.",
        "Read the 'Terms & Conditions' before proceeding.",
        "COVID-19 pandemic has changed the world.",
        "Visit us at www.openai.com for more information.",
        "The meeting is scheduled for next Monday, March 15th.",
        "She scored 98.6% on her exam.",
        "It's a well-known fact that water freezes at 0°C.",
        "He exclaimed, 'Wow! That's amazing!'",
        "Please RSVP by July 20th.",
        "The stock price increased by 12.5% yesterday.",
        "Is it pronounced 'GIF' or 'JIF'?",
        "The package will arrive between 9:00 AM and 5:00 PM.",
        "They live in a three-bedroom, two-bath house.",
        "She has a master's degree in computer science.",
        "The URL is https://www.example.com/page?query=test.",
        "I need to schedule a 30-minute meeting.",
        "He won the 100-meter dash in 9.58 seconds.",
        "Please don't forget to back up your data.",
        "The recipe calls for 2 cups of flour and 1.5 teaspoons of salt.",
        "They're going to the movies tonight.",
        "It's raining cats and dogs outside.",
        "The CEO's speech was both inspiring and informative.",
        "She received a 4.0 GPA this semester.",
        "The ISBN number for the book is 978-3-16-148410-0.",
        "Can you pass the salt, please?",
        "The concert starts at 8 PM sharp.",
        "He drives a Tesla Model S.",
        "The cost is $19.99 after a 20% discount.",
        "They've been married for twenty years.",
        "The flight number is AA1234 departing at 6:45 AM.",
        "It's a state-of-the-art facility.",
        "Please refer to section 5.2 of the manual.",
        "The temperature dropped to -5 degrees last night.",
        "She’s an avid reader of science fiction novels.",
        "The package weighs 2.5 kilograms.",
        "He's a well-respected member of the community.",
        "Their favorite sports are basketball and soccer.",
        "Remember to update your antivirus software regularly."
    ]

    for phrase in phrases:
        spoken = english_cleaners(phrase)
        print(f"Original: {phrase}\nSpoken: {spoken}\n")
