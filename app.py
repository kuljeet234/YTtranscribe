import streamlit as st
import os
from ytTranscribe import download_youtube_video, transcribe_video, save_transcription_to_json, search_transcriptions_by_context

# Define paths for download folder and transcription file
output_folder = "download"
transcription_file = "video_transcriptions.json"

# Function to clear the library
def clear_library():
    # Delete all files in the download folder
    if os.path.exists(output_folder):
        for file in os.listdir(output_folder):
            file_path = os.path.join(output_folder, file)
            if os.path.isfile(file_path):
                os.remove(file_path)
    
    # Clear the content of the transcription file
    if os.path.exists(transcription_file):
        with open(transcription_file, "w") as f:
            f.write("{}")  # Reset JSON file to an empty dictionary

# Streamlit app layout
st.title("YouTube Video Search Based on Transcriptions")
st.write("Enter a text to find videos that match based on their transcriptions.")

# Input text from user for searching
input_text = st.text_area("Enter text to search", "")

if st.button("Search"):
    if input_text.strip() == "":
        st.error("Please enter some text to search.")
    else:
        # Search for matching videos
        matching_videos = search_transcriptions_by_context(input_text)
        
        if matching_videos:
            st.success(f"Found {len(matching_videos)} matching videos:")
            for link in matching_videos:
                st.write(link)
        else:
            st.warning("No matching videos found.")

# Divider
st.write("---")

# Create an expandable "Add Video" section
with st.expander("Add Video", expanded=False):
    # Input field for YouTube video link
    video_link = st.text_input("Enter YouTube video link", "")

    if st.button("Download and Transcribe"):
        if video_link.strip() == "":
            st.error("Please enter a valid YouTube video link.")
        else:
            # Display a spinner while the video is being downloaded and transcribed
            with st.spinner("Downloading and transcribing..."):
                try:
                    # Download the YouTube video
                    video_path, video_title = download_youtube_video(video_link, output_folder)

                    # Verify if the file exists
                    if os.path.exists(video_path):
                        # Transcribe the video
                        transcription = transcribe_video(video_path)

                        # Save the transcription to a JSON file
                        save_transcription_to_json(video_link, transcription)

                        # Display success message once done
                        st.success(f"Transcription completed for '{video_title}'!")
                    else:
                        st.error("Error: The video file was not found.")
                except Exception as e:
                    st.error(f"An error occurred: {e}")

# Add the "Clear Library" button with proper confirmation
if st.button("Clear Library"):
    clear_library()
    st.success("Library cleared! All downloaded videos and transcriptions have been deleted.")
