# This script processes Evernote .enex files
# and converts them into Markdown files,
# organizing them into separate folders.
# it uses the following github repo: https://github.com/wormi4ok/evernote2md

# to install the packages from the repo, run the following command: brew install evernote2md
# separately, you'll need to download all the .enex files from your Evernote before running the script below.

for file in ./EvernoteLinks/*.enex; do
  folder_name=$(basename "$file" .enex)
  output_folder="./MarkdownNotes/$folder_name"
  mkdir -p "$output_folder"
  evernote2md "$file" "$output_folder" --tagTemplate "evernote, {{tag}}"
  
  # Post-process markdown files
  find "$output_folder" -name "*.md" -exec sh -c '
    for mdfile in "$0"/*; do
      # Replace Evernote links with markdown links
      sed -i "" -E '\''
        # Handle internal Evernote links
        s|evernote:///view/[0-9]*/[^/]*/[^/]*/[^/]*/([^/]*)|[[note-\1]]|g
        # Handle web Evernote links
        s|https://www\.evernote\.com/client/web#/note/[^[:space:]]*|[[note]]|g
      '\'' "$mdfile"
    done
  ' "$output_folder" \;
done