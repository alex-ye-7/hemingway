# Scraping Hemingway text 

import requests

# Dictionary of Hemingway works and their Project Gutenberg IDs
hemingway_works = {
    "The_Sun_Also_Rises": "67138"
    # "In_Our_Time": "61085"
}

def get_cleaned(book_id):
    url = f"https://www.gutenberg.org/cache/epub/{book_id}/pg{book_id}.txt"
    response = requests.get(url)
    if response.status_code != 200:
        return None

    full_text = response.text
    # Gutenberg uses variations of these phrases
    start_markers = [
        "*** START OF THE PROJECT GUTENBERG EBOOK",
        "*** START OF THIS PROJECT GUTENBERG EBOOK"
    ]
    end_markers = [
        "*** END OF THE PROJECT GUTENBERG EBOOK",
        "*** END OF THIS PROJECT GUTENBERG EBOOK"
    ]

    # Find the beginning of the actual text
    start_index = -1
    for marker in start_markers:
        pos = full_text.find(marker)
        if pos != -1:
            start_index = full_text.find("\n", pos + len(marker)) + 1
            break

    # Find the end of the actual text
    end_index = -1
    for marker in end_markers:
        pos = full_text.find(marker)
        if pos != -1:
            end_index = pos
            break

    # Slice the text
    if start_index != -1 and end_index != -1:
        pure_text = full_text[start_index:end_index].strip()
        return pure_text
    else:
        return "Could not isolate text. Markers not found."

for key, value in hemingway_works.items():
    clean_text = get_cleaned(value)
    if clean_text:
        with open(f"data/{key}.txt", "w", encoding="utf-8") as f:
            f.write(clean_text)
        print(f"Cleaned text extracted and saved to data/{key}.txt")