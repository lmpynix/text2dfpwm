import os
import logging

import numpy as np

import torch
from parler_tts import ParlerTTSForConditionalGeneration
from transformers import AutoTokenizer

import dfpwm
import librosa

from fastapi import FastAPI, BackgroundTasks, Response, status
from fastapi.responses import StreamingResponse

from diskcache import Cache

# Default directory for cache files.
CACHE_PATH = "tts_cache"

VOICE_DESCRIPTION = "A male speaker with a very low pitched voice speaks at a very slow speed.  The recording is very clear with no background noise, with the speaker very close up."

# Configure logging
log = logging.getLogger(__name__)
if os.getenv("TTS_DEBUG", None) is not None:
    logging.basicConfig(level=logging.DEBUG)
else:
    logging.basicConfig(level=logging.INFO)

# Configure torch.
torch.set_num_threads(12)
torch.set_num_interop_threads(12)

# Set up static parts of TTS model.
model = ParlerTTSForConditionalGeneration.from_pretrained("parler-tts/parler-tts-mini-v1").to("cpu")
tokenizer = AutoTokenizer.from_pretrained("parler-tts/parler-tts-mini-v1")
description_input_ids = tokenizer(VOICE_DESCRIPTION, return_tensors="pt").input_ids.to("cpu")

# Audio render function, blocking.
def render_audio(model, tokenizer, desc, prompt, cache):
    '''
    Render the prompt using the given model, tokenizer, and voice description, inserting the result into the given cache.
    '''

    # Use ML model to generate voice audio, squeeze to a numpy array of samples.
    prompt_input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to("cpu")
    generation = model.generate(input_ids=desc, prompt_input_ids=prompt_input_ids)
    audio_a = generation.cpu().numpy().squeeze()

    # Use DFPWM library to convert array to DFPWM after resampling.
    original_sample_rate = model.config.sampling_rate
    resampled = librosa.resample(audio_a, orig_sr=original_sample_rate, target_sr=dfpwm.SAMPLE_RATE).astype(np.float32)
    audio_d = dfpwm.compressor(resampled)
    cache[prompt] = audio_d

cache = Cache(os.getenv("TTS_CACHE_PATH", CACHE_PATH))
log.info("Initialized.")
app = FastAPI()

@app.get("/render/", status_code=200)
async def tts_endpoint(prompt: str, bg_tasks: BackgroundTasks):
    # De-encode query string
    tts_str = prompt.replace('+', ' ')

    if tts_str in cache:
        if cache[tts_str] is not None:
            # Result is cached.
            log.debug("HIT: %s", tts_str)
            return Response(content=cache[tts_str].tobytes(), media_type="audio/dfpwm")
        else:
            # Result has been requested, but rendering isn't complete.
            log.debug("NOT DONE YET: %s", tts_str)
            return Response(status_code=status.HTTP_202_ACCEPTED)
    else:
        # Result is not cached, which means we need to generate it.
        log.debug("MISS: %s", tts_str)

        # Add empty entry in cache so we know processing has started.
        cache.add(tts_str, None)

        # Spawn background task.
        bg_tasks.add_task(render_audio, model, tokenizer, description_input_ids, tts_str, cache)

        return Response(status_code=status.HTTP_202_ACCEPTED)
