import os
import re
from pathlib import Path

# Directory containing your converted markdown files
base_dir = Path("./MarkdownNotes")

# Process all markdown files in all subdirectories
for md_file in base_dir.rglob("*.md"):
    try:
        # Read the content
        content = md_file.read_text(encoding='utf-8')
        
        # Replace Evernote internal links
        content = re.sub(
            r'evernote:///view/[0-9]*/[^/]*/[^/]*/[^/]*/([^/]*)',
            r'[[note-\1]]',
            content
        )
        
        # Replace Evernote web links
        content = re.sub(
            r'https://www\.evernote\.com/client/web#/note/[^\s]*',
            r'[[note]]',
            content
        )
        
        # Write the modified content back
        md_file.write_text(content, encoding='utf-8')
        print(f"Processed: {md_file}")
        
    except Exception as e:
        print(f"Error processing {md_file}: {str(e)}")