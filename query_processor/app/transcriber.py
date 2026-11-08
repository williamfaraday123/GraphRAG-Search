"""
Speech-to-text for audio uploads, using faster-whisper (local, CPU-friendly Whisper).

Transcription runs in a separate worker process. Its native libraries (CTranslate2,
onnxruntime) can crash the whole process on a broken setup, e.g. an outdated Visual C++
runtime on Windows. Isolating them means a crash fails the upload with a clear message
instead of taking down the gRPC server and every search in progress. The worker keeps
the model loaded between uploads; the first load downloads it from Hugging Face
(~150 MB for "base") into the HF cache.

Configuration (environment variables):
  WHISPER_MODEL         tiny | base | small | medium | large-v3   (default: base)
  WHISPER_DEVICE        cpu | cuda | auto                         (default: cpu)
  WHISPER_COMPUTE_TYPE  int8 | float16 | float32                  (default: int8, fastest on CPU)
"""
import io
import logging
import os
import threading
import time
from concurrent.futures import ProcessPoolExecutor
from concurrent.futures.process import BrokenProcessPool

logger = logging.getLogger(__name__)

# Start a new paragraph when the speaker pauses this long, so the chunker
# gets natural split points instead of one wall of text.
PARAGRAPH_PAUSE_SECONDS = 1.5

CRASH_MESSAGE = (
    "Audio transcription crashed. On Windows this usually means the Microsoft Visual C++ "
    "Redistributable is outdated; install the latest x64 version from "
    "https://aka.ms/vs/17/release/vc_redist.x64.exe and restart the query processor"
)

_pool = None
_pool_lock = threading.Lock()


def transcribe(audio: bytes) -> str:
    """Returns the transcript of an audio file. Raises ValueError with a user-facing
    message if the audio can't be decoded, contains no speech, or the worker crashes."""
    global _pool
    with _pool_lock:  # one transcription at a time; Whisper already uses every CPU core
        if _pool is None:
            _pool = ProcessPoolExecutor(max_workers=1)
        try:
            return _pool.submit(_transcribe_in_worker, audio).result()
        except BrokenProcessPool as exc:
            logger.error("Transcription worker crashed; it will be restarted on the next upload")
            _pool = None  # a broken pool can't be reused
            raise ValueError(CRASH_MESSAGE) from exc


# --- Everything below runs inside the worker process ---

_model = None


def _get_model():
    global _model
    if _model is None:
        from faster_whisper import WhisperModel

        name = os.getenv("WHISPER_MODEL", "base")
        device = os.getenv("WHISPER_DEVICE", "cpu")
        compute_type = os.getenv("WHISPER_COMPUTE_TYPE", "int8")
        logger.info("Loading Whisper model '%s' on %s (%s)...", name, device, compute_type)
        _model = WhisperModel(name, device=device, compute_type=compute_type)
    return _model


def _transcribe_in_worker(audio: bytes) -> str:
    model = _get_model()
    started = time.time()
    try:
        # vad_filter skips silence/music, which avoids Whisper hallucinating text there
        segments, info = model.transcribe(io.BytesIO(audio), vad_filter=True)
        paragraphs, current, last_end = [], [], None
        for seg in segments:  # generator: decoding happens while iterating
            text = seg.text.strip()
            if not text:
                continue
            if last_end is not None and seg.start - last_end >= PARAGRAPH_PAUSE_SECONDS and current:
                paragraphs.append(" ".join(current))
                current = []
            current.append(text)
            last_end = seg.end
        if current:
            paragraphs.append(" ".join(current))
    except Exception as exc:
        logger.warning("Could not transcribe audio: %s", exc)
        raise ValueError("Could not read this audio file; it may be damaged or not a real MP3") from exc

    transcript = "\n\n".join(paragraphs)
    logger.info(
        "Transcribed %.0fs of %s audio in %.0fs (%d chars)",
        info.duration, info.language, time.time() - started, len(transcript),
    )
    if not transcript.strip():
        raise ValueError("No speech was found in this audio file")
    return transcript
