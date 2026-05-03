"""
Simplified Shakespeare preprocessing for federated learning.
Works with modern Project Gutenberg text format.

Usage:
  python src/preprocess_shakespeare_v2.py data/raw/shakespeare.txt data/federation/
"""

import json
import os
import re
import sys
from collections import defaultdict


def preprocess_shakespeare(input_file, output_dir):
    """Extract character dialogues from Shakespeare text."""
    
    print("Reading Shakespeare file...")
    with open(input_file, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Split by play titles (ALL CAPS lines followed by newlines)
    # Plays start with titles like "ALL'S WELL THAT ENDS WELL"
    play_pattern = r'^([A-Z][A-Z\'\s]+)$\n\n'
    plays_split = re.split(play_pattern, content, flags=re.MULTILINE)
    
    character_dialogues = defaultdict(list)
    plays_metadata = {}
    
    # Process pairs of (title, text)
    for i in range(1, len(plays_split), 2):
        if i + 1 >= len(plays_split):
            break
        
        play_title = plays_split[i].strip()
        play_text = plays_split[i + 1]
        
        if not play_title or len(play_title) < 5:
            continue
        
        print(f"Processing: {play_title[:50]}")
        
        # Extract character names and their lines
        # Pattern: Character name in ALL CAPS on its own line, followed by dialogue
        # This regex looks for CAPS followed by .: and then text
        char_pattern = r'^([A-Z][A-Z\s]+?)\.?\n((?:(?!^[A-Z])[^\n]|\n(?!  ))*)'
        
        matches = re.finditer(char_pattern, play_text, re.MULTILINE | re.IGNORECASE)
        
        for match in matches:
            char_name = match.group(1).strip().upper()
            dialogue = match.group(2).strip()
            
            if char_name and dialogue and len(dialogue) > 10:
                # Clean up the dialogue
                dialogue = ' '.join(dialogue.split())
                character_dialogues[char_name].append(dialogue)
                plays_metadata[char_name] = play_title
    
    print(f"\nFound {len(character_dialogues)} unique characters")
    
    # Write character files
    os.makedirs(output_dir, exist_ok=True)
    split_dir = os.path.join(output_dir, 'users_split_by_pearsons')
    os.makedirs(split_dir, exist_ok=True)
    
    valid_chars = 0
    for char_name, dialogues in character_dialogues.items():
        if len(dialogues) >= 5:  # Minimum dialogue threshold
            filename = os.path.join(split_dir, f"{char_name.replace(' ', '_')}.txt")
            with open(filename, 'w', encoding='utf-8') as f:
                f.write('\n'.join(dialogues))
            valid_chars += 1
    
    # Save metadata
    metadata = {name: plays_metadata[name] for name in character_dialogues.keys()}
    with open(os.path.join(output_dir, 'users_and_plays.json'), 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"Wrote {valid_chars} character files to {split_dir}/")
    print(f"Metadata saved to {os.path.join(output_dir, 'users_and_plays.json')}")


if __name__ == '__main__':
    if len(sys.argv) < 3:
        print("Usage: python preprocess_shakespeare_v2.py <input_file> <output_dir>")
        sys.exit(1)
    
    preprocess_shakespeare(sys.argv[1], sys.argv[2])
