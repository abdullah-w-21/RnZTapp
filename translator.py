import os
import tempfile
import streamlit as st
import speech_recognition as sr
from moviepy.editor import VideoFileClip, AudioFileClip
from pydub import AudioSegment
import google.generativeai as genai
import numpy as np
import librosa
import edge_tts
import asyncio
import base64

from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Get API key from environment variable
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
genai.configure(api_key=GOOGLE_API_KEY)




# Function to load and display the logo
def add_logo():
    logo_path = "logo.png"
    if os.path.exists(logo_path):
        with open(logo_path, "rb") as f:
            data = f.read()
            encoded = base64.b64encode(data).decode()

        st.markdown(
            f"""
            <style>
                [data-testid="stSidebarNav"] {{
                    background-image: url("data:image/png;base64,{encoded}");
                    background-repeat: no-repeat;
                    background-position: 20px 20px;
                    background-size: 200px auto;
                    padding-top: 120px;
                }}
            </style>
            """,
            unsafe_allow_html=True,
        )


def check_video_duration(video_path):
    """Check if video duration is within limit"""
    try:
        with VideoFileClip(video_path) as video:
            duration = video.duration
            return duration <= 90  # 90 seconds = 1.5 minutes
    except Exception as e:
        st.error(f"Error checking video duration: {str(e)}")
        return False


async def generate_speech(text):
    """Generate realistic male voice speech using edge-tts"""
    try:
        voice = "en-US-GuyNeural"
        temp_audio_path = tempfile.mktemp(suffix=".mp3")

        communicate = edge_tts.Communicate(text, voice)
        await communicate.save(temp_audio_path)

        return temp_audio_path

    except Exception as e:
        st.error(f"Error in speech generation: {str(e)}")
        return None


def extract_audio(video_path):
    """Extract audio from video and analyze pitch patterns"""
    try:
        video = VideoFileClip(video_path)
        audio = video.audio
        temp_audio_path = tempfile.mktemp(suffix=".wav")
        audio.write_audiofile(temp_audio_path, codec='pcm_s16le')

        y, sr = librosa.load(temp_audio_path)
        pitches, magnitudes = librosa.piptrack(y=y, sr=sr)
        pitch_patterns = []

        for i in range(0, len(y), sr):
            segment = pitches[:, i:i + sr]
            if segment.size > 0:
                avg_pitch = np.mean(segment[segment > 0]) if np.any(segment > 0) else 0
                pitch_patterns.append(avg_pitch)

        return temp_audio_path, pitch_patterns

    except Exception as e:
        st.error(f"Error in audio extraction: {str(e)}")
        return None, None


def recognize_speech(audio_path):
    """Convert Urdu speech to text using Google Speech Recognition"""
    recognizer = sr.Recognizer()

    try:
        audio = AudioSegment.from_wav(audio_path)
        chunk_length = 30 * 1000
        chunks = [audio[i:i + chunk_length] for i in range(0, len(audio), chunk_length)]
        transcribed_text = []

        with st.spinner('Recognizing speech... This may take a while.'):
            for i, chunk in enumerate(chunks):
                chunk_path = tempfile.mktemp(suffix=".wav")
                chunk.export(chunk_path, format="wav")

                with sr.AudioFile(chunk_path) as source:
                    audio_data = recognizer.record(source)
                    try:
                        text = recognizer.recognize_google(audio_data, language="ur-PK")
                        transcribed_text.append(text)
                        st.text(f"Chunk {i + 1} recognized")
                    except sr.UnknownValueError:
                        st.warning(f"Could not understand audio in chunk {i + 1}")
                    except sr.RequestError as e:
                        st.error(f"Error with the speech recognition service: {e}")

                os.remove(chunk_path)

        return " ".join(transcribed_text)

    except Exception as e:
        st.error(f"Error in speech recognition: {str(e)}")
        return None


async def process_video(video_path):
    """Main video processing function"""
    try:
        progress_bar = st.progress(0)
        status_text = st.empty()

        # Process video
        with VideoFileClip(video_path) as video:
            # Extract audio
            status_text.text("Extracting audio and analyzing pitch patterns...")
            audio_path, pitch_patterns = extract_audio(video_path)
            if not audio_path:
                return None
            progress_bar.progress(20)

            # Recognize speech
            status_text.text("Recognizing Urdu speech...")
            urdu_text = recognize_speech(audio_path)
            if not urdu_text:
                return None
            progress_bar.progress(40)

            # Translate
            status_text.text("Translating to English...")
            model = genai.GenerativeModel('gemini-pro')
            response = model.generate_content(f"Translate this Urdu text to English: {urdu_text}")
            english_text = response.text
            progress_bar.progress(60)

            # Generate speech
            status_text.text("Generating English speech...")
            english_audio_path = await generate_speech(english_text)
            if not english_audio_path:
                return None
            progress_bar.progress(80)

            # Create final video
            status_text.text("Creating final video...")
            output_path = os.path.splitext(video_path)[0] + "_translated.mp4"

            # Combine video with new audio
            final_video = video.set_audio(AudioFileClip(english_audio_path))
            final_video.write_videofile(output_path, codec='libx264', audio_codec='aac')
            final_video.close()

            # Clean up
            for file_path in [audio_path, english_audio_path]:
                try:
                    if os.path.exists(file_path):
                        os.remove(file_path)
                except:
                    pass

            progress_bar.progress(100)
            status_text.text("Video processing complete!")
            return output_path

    except Exception as e:
        st.error(f"Error processing video: {str(e)}")
        return None


def main():
    # Set page config
    st.set_page_config(
        page_title="Advanced Urdu Video Translator",
        page_icon="🎥",
        layout="wide"
    )

    # Add logo
    st.image("logrnz.png", width=150)

    # Custom CSS
    st.markdown("""
        <style>
        .main-header {
            text-align: center;
            padding: 20px;
        }
        .company-name {
            color: #2e7d32;
            font-size: 1.2em;
            font-weight: bold;
        }
        .error-message {
            color: #ff0000;
            font-weight: bold;
            padding: 10px;
            border-radius: 5px;
            background-color: #ffe6e6;
        }
        </style>
    """, unsafe_allow_html=True)

    # Header
    st.markdown('<div class="main-header">', unsafe_allow_html=True)
    st.title("Advanced Urdu Video Translator")
    st.markdown('<p class="company-name">Powered by RnZ Technologies</p>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

    st.write("Upload a video with Urdu speech to translate it to English with matching voice characteristics.")
    st.warning("Note: Video duration must be 1.5 minutes or less")

    uploaded_file = st.file_uploader("Choose a video file", type=["mp4", "avi", "mov"])

    if uploaded_file is not None:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp_file:
            tmp_file.write(uploaded_file.read())
            tmp_file_path = tmp_file.name

        # Check video duration
        if not check_video_duration(tmp_file_path):
            st.markdown('<p class="error-message">Error: Video duration must be 1.5 minutes or less</p>',
                        unsafe_allow_html=True)
            try:
                os.remove(tmp_file_path)
            except:
                pass
        else:
            st.video(uploaded_file)

            if st.button("Process Video"):
                try:
                    output_path = asyncio.run(process_video(tmp_file_path))

                    if output_path and os.path.exists(output_path):
                        st.success("Video processing complete!")
                        st.video(output_path)

                        with open(output_path, "rb") as file:
                            btn = st.download_button(
                                label="Download processed video",
                                data=file,
                                file_name="processed_video.mp4",
                                mime="video/mp4"
                            )

                        try:
                            os.remove(output_path)
                        except:
                            pass
                finally:
                    try:
                        if os.path.exists(tmp_file_path):
                            os.remove(tmp_file_path)
                    except:
                        pass

    # Footer
    st.markdown("---")
    st.markdown("© 2024 RnZ Technologies. All rights reserved.")


if __name__ == "__main__":
    main()