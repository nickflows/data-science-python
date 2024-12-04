import evernote.edam.userstore.constants as UserStoreConstants
import evernote.edam.notestore.NoteStore as NoteStore
import os

# Authenticate with your Evernote credentials
from evernote.api.client import EvernoteClient

# Replace these with your Evernote API key and secret
consumer_key = 'your_consumer_key'
consumer_secret = 'your_consumer_secret'
auth_token = 'your_auth_token'

client = EvernoteClient(token=auth_token, sandbox=False)

# Get the NoteStore object
note_store = client.get_note_store()

# Fetch all notebooks
notebooks = note_store.listNotebooks()

# Base directory for exported notebooks
base_export_path = "Evernote_Export"
os.makedirs(base_export_path, exist_ok=True)

# Create folders for stacks and organize notebooks
for notebook in notebooks:
    notebook_name = notebook.name
    stack_name = notebook.stack if notebook.stack else "Unstacked"

    # Create a folder for the stack
    stack_path = os.path.join(base_export_path, stack_name)
    os.makedirs(stack_path, exist_ok=True)

    # Export notebook (Here, we simply create a placeholder file for demonstration)
    # Replace this with actual notebook export logic if needed
    notebook_file_path = os.path.join(stack_path, f"{notebook_name}.txt")

    with open(notebook_file_path, "w") as f:
        f.write(f"Notebook: {notebook_name}\nStack: {stack_name}\n")
        f.write("This is a placeholder for notebook content.\n")
        # You can use note_store.findNotes() to fetch individual notes from a notebook

    print(f"Exported notebook '{notebook_name}' to stack '{stack_name}'")

print(f"All notebooks have been exported to {base_export_path}.")
