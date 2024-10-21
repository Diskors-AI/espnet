import re
import os
from praatio import textgrid

# Precompile all regular expressions:
_whitespace_re = re.compile(r"\s+")
_punctuation_re = re.compile(r"([,;:.!?()])")
_quotes_re = re.compile(r'["]')
_sound_markers_re = re.compile(r"\{[^}]+\}")
_mid_word_quotes_re = re.compile(r'(\w)-"([^"]+)"')
_parentheses_brackets_re = re.compile(r"[\[\]\(\)]")


# Maltese-specific cleaner
def clean(text):
    """
    Maltese-specific text cleaning pipeline that retains semicolons, colons, hyphens,
    and quotation marks. The cleaner assumes that numbers and abbreviations are already expanded.
    """

    # It's important that these two functions are called before the data is manipulated further
    # since their regular expressions might not match the intended words and phrases.
    text = remove_sound_markers(text)
    text = remove_mid_word_quotes(text)

    text = remove_parentheses_and_brackets(text)
    text = separate_punctuation(text)
    text = collapse_whitespace(text)
    text = lowercase(text)

    return text


def separate_punctuation(text):
    """
    Ensure punctuation like periods, commas, semicolons, colons, etc., are not attached to words.
    This keeps the punctuation distinct, allowing the model to better learn prosody.
    """

    # Adding spaces around punctuation marks to ensure separation
    # This keeps semicolons, colons, and quotes but separates them with spaces.
    text = _punctuation_re.sub(r" \1 ", text)

    # Ensure double quotes are also spaced properly
    text = _quotes_re.sub(r' " ', text)

    # Collapse any additional whitespace from adding spaces
    text = collapse_whitespace(text)

    return text


def collapse_whitespace(text):
    """Collapses multiple spaces into a single space."""
    return _whitespace_re.sub(" ", text).strip()


def lowercase(text):
    """Lowercases the text."""
    return text.lower()


def remove_sound_markers(text):
    return _sound_markers_re.sub("", text)


def remove_mid_word_quotes(text):
    # Remove double quotes only when they appear in the middle of a unit, e.g., between a hyphen or surrounding an article
    # like, for example, il-"Materjal Sedizzjuż".
    return _mid_word_quotes_re.sub(r"\1-\2", text)

def remove_parentheses_and_brackets(text):
    """
    Removes only parentheses and brackets from the text.

    Parameters:
    - text (str): The input text to be cleaned.

    Returns:
    - str: The cleaned text with parentheses and brackets removed.
    """
    return _parentheses_brackets_re.sub("", text)


def test_with_textgrids(folder_path, output_file_path):
    # Open the output file in write mode
    with open(output_file_path, "w", encoding="utf-8") as output_file:
        # Loop through all TextGrid files in the folder
        for filename in os.listdir(folder_path):
            if filename.endswith(".TextGrid"):
                textgrid_path = os.path.join(folder_path, filename)

                # Load the TextGrid file
                tg = textgrid.openTextgrid(textgrid_path, includeEmptyIntervals=True)

                # Get the first tier (assuming the first tier is the one you want)
                first_tier = tg.getTier(
                    tg.tierNames[0],
                )

                # Loop through the intervals in the first tier
                for interval in first_tier.entries:
                    original_text = interval.label

                    if original_text.strip() == "":
                        continue

                    # Clean the text with the Maltese cleaner
                    cleaned_text = clean(original_text)

                    # Write original and cleaned text separated by a tab to the file
                    output_file.write(f"{original_text}\t{cleaned_text}\n")

    print(f"Processing complete. Output saved to {output_file_path}")
