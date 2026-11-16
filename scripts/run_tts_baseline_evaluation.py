#!/usr/bin/env python3
"""
MWIZERWA AI — Baseline TTS Evaluation Script
Generates reproducible baseline audio using the existing, unchanged KinyaFlex
checkpoint across all 18 benchmark sentences and 3 speaker IDs (0, 1, 2).

Outputs 54 WAV files (24 kHz, mono, 16-bit PCM) and a comprehensive manifest.json
under outputs/tts_baseline_evaluation/.
"""

import datetime
import hashlib
import json
import os
import sys
import time
import wave
from pathlib import Path
from typing import Dict, List, Any

import numpy as np
import torch

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Also ensure DeepKIN-AgAI is available
DEEPKIN_DIR = PROJECT_ROOT / "ac-ai-models" / "DeepKIN-AgAI"
if str(DEEPKIN_DIR) not in sys.path:
    sys.path.insert(0, str(DEEPKIN_DIR))

from deepkin.data.kinya_norm import text_to_sequence
from deepkin.models.flex_tts import FlexKinyaTTS
from deepkin.modules.tts_commons import intersperse

# ==============================================================================
# Configuration
# ==============================================================================

MODEL_PATH = PROJECT_ROOT / "models" / "kinya-flex-tts" / "kinya_flex_tts_base_trained.pt"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "tts_baseline_evaluation"
MANIFEST_PATH = OUTPUT_DIR / "manifest.json"

SAMPLE_RATE = 24000
SPEED = 1.0

SPEAKERS = {
    0: "female_1",
    1: "female_2",
    2: "male",
}

BENCHMARK_SENTENCES: List[Dict[str, Any]] = [
    {
        "id": "01",
        "category": "difficult_consonant_clusters",
        "text": "Umwana yanyuze mu nshyamba arwaye impyiko n’ibicurane bikabije.",
        "description": "Evaluates consonant clusters: nshy, mpy, kib.",
    },
    {
        "id": "02",
        "category": "difficult_consonant_clusters",
        "text": "Ibikoresho by’ubutsimbaraye byifashishwa mu buhinzi bwa kijyambere.",
        "description": "Evaluates consonant clusters: by, ts, mb, jy.",
    },
    {
        "id": "03",
        "category": "difficult_consonant_clusters",
        "text": "Yambaye imyambaro ibengerana maze arishyuza amafaranga ye yose.",
        "description": "Evaluates consonant clusters: my, beng, shy, ry.",
    },
    {
        "id": "04",
        "category": "short_acknowledgement",
        "text": "Ndakumva neza, kandi humura ndahari ngo ngufashe.",
        "description": "Active listening, reassuring presence.",
    },
    {
        "id": "05",
        "category": "short_acknowledgement",
        "text": "Nta kibazo na gito, fata umwanya wawe ushire ubwoba.",
        "description": "Calming, anxiety reduction, deliberate pacing.",
    },
    {
        "id": "06",
        "category": "short_acknowledgement",
        "text": "Ndumva ububabare ufite, kandi birumvikana rwose.",
        "description": "Warm validation, empathetic confirmation.",
    },
    {
        "id": "07",
        "category": "emotional_validation",
        "text": "Birababaje cyane kubona unyura muri ibyo bihe bikomeye kandi biteye agahinda.",
        "description": "Solemn emotional acknowledgement, compassionate cadence.",
    },
    {
        "id": "08",
        "category": "emotional_validation",
        "text": "Ibyiyumvo byawe byose bifite agaciro gakomeye, kandi nturi wenyine muri uru rugendo.",
        "description": "Deep validation, emotional warmth, reassurance.",
    },
    {
        "id": "09",
        "category": "emotional_validation",
        "text": "N’ubwo wumva wikoreye umutwaro uremereye, hari ibyiringiro by’uko ejo hazaba heza.",
        "description": "Supportive comfort, gentle encouragement.",
    },
    {
        "id": "10",
        "category": "gentle_explanation",
        "text": "Guhangayika kenshi bitera umubiri kunanirwa, ariko gushaka ubufasha ni intambwe nziza.",
        "description": "Psychoeducation, medical clarity with gentle delivery.",
    },
    {
        "id": "11",
        "category": "gentle_explanation",
        "text": "Iyo umutima urebye ibintu byose icyarimwe, birafasha gufata akanya ugatuza gato.",
        "description": "Cognitive grounding, reflective pacing.",
    },
    {
        "id": "12",
        "category": "gentle_guidance_breathing",
        "text": "Fata umwuka mwiza witonze, winjize umwuka mu bihaha, hanyuma uwusohore gahoro.",
        "description": "Somatic breathing guidance, slow rhythmic cadence.",
    },
    {
        "id": "13",
        "category": "gentle_guidance_breathing",
        "text": "Tugiye kubirebera hamwe buhoro buhoro, nta gitutu na gito uri gushyirwaho.",
        "description": "Grounding and reassurance, gentle tempo.",
    },
    {
        "id": "14",
        "category": "open_question",
        "text": "Ni iki kiguhangayikishije cyane muri uyu mwanya wifuza ko twaganiraho?",
        "description": "Open therapeutic question, natural interrogative inflection.",
    },
    {
        "id": "15",
        "category": "open_question",
        "text": "Waba wumva hari ikintu runaka kigufasha kuruhuka iyo wumva uguye umutima?",
        "description": "Gentle inquiry into coping mechanisms.",
    },
    {
        "id": "16",
        "category": "reassuring_closure",
        "text": "Komeza wihangane, buri ntambwe yose uteye uyu munsi ifite agaciro gakomeye.",
        "description": "Supportive affirmation, closure encouragement.",
    },
    {
        "id": "17",
        "category": "reassuring_closure",
        "text": "Niba wumva unaniwe, dushobora gufata akaruhuko gato tukazakomeza ikiganiro nyuma.",
        "description": "Compassionate boundary setting, client autonomy.",
    },
    {
        "id": "18",
        "category": "complex_multi_clause",
        "text": "Nubwo bitoroshye gusobanura ibyo urimo kunyuramo, kuba watangiye kubivuga ni ikimenyetso cy’imbaraga zikomeye wifitemo.",
        "description": "Complex syntactic structure, multi-clause prosodic flow.",
    },
]


def compute_sha256(filepath: Path) -> str:
    """Computes SHA-256 hash of a file."""
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(8192 * 1024):
            h.update(chunk)
    return h.hexdigest()


def verify_wav_file(filepath: Path, expected_sample_rate: int = 24000) -> Dict[str, Any]:
    """Programmatically verifies that a WAV file conforms to expected PCM specifications."""
    with wave.open(str(filepath), "rb") as wf:
        nchannels = wf.getnchannels()
        sampwidth = wf.getsampwidth()
        framerate = wf.getframerate()
        nframes = wf.getnframes()
        duration = nframes / float(framerate) if framerate > 0 else 0.0

    return {
        "valid": (nchannels == 1 and sampwidth == 2 and framerate == expected_sample_rate and nframes > 0),
        "nchannels": nchannels,
        "sampwidth": sampwidth,
        "bit_depth": sampwidth * 8,
        "framerate": framerate,
        "nframes": nframes,
        "duration_seconds": round(duration, 4),
    }


def main():
    print("=" * 70)
    print("MWIZERWA AI — Baseline TTS Evaluation (18 Sentences × 3 Speakers)")
    print("=" * 70)

    # 1. Check Model Path
    if not MODEL_PATH.exists():
        print(f"Error: Model checkpoint not found at: {MODEL_PATH}")
        sys.exit(1)

    print(f"Model Checkpoint: {MODEL_PATH}")
    print("Computing SHA-256 hash of checkpoint...")
    checkpoint_sha256 = compute_sha256(MODEL_PATH)
    print(f"Checkpoint SHA-256: {checkpoint_sha256}")

    # 2. Check and Create Output Directory
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    existing_wavs = list(OUTPUT_DIR.glob("*.wav"))
    if existing_wavs:
        print(f"[Notice] Output directory already contains {len(existing_wavs)} WAV files.")

    # 3. Device Selection
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Inference Device: {device}")

    # 4. Load Model
    print("Loading FlexKinyaTTS model...")
    t_load_start = time.perf_counter()
    kinya_tts = FlexKinyaTTS.from_pretrained(device, str(MODEL_PATH))
    kinya_tts.eval()
    t_load_elapsed = time.perf_counter() - t_load_start
    print(f"Model loaded successfully in {t_load_elapsed:.2f} seconds.")
    print("-" * 70)

    # 5. Evaluation Loop
    total_planned = len(BENCHMARK_SENTENCES) * len(SPEAKERS)
    print(f"Starting generation: {len(BENCHMARK_SENTENCES)} sentences × {len(SPEAKERS)} speakers = {total_planned} audio files.")
    print("-" * 70)

    manifest_entries: List[Dict[str, Any]] = []
    success_count = 0
    failure_count = 0

    run_start_iso = datetime.datetime.now(datetime.timezone.utc).isoformat()
    total_synthesis_time = 0.0
    total_audio_duration = 0.0

    for s_idx, sent in enumerate(BENCHMARK_SENTENCES, start=1):
        sent_id = sent["id"]
        category = sent["category"]
        text = sent["text"]
        description = sent["description"]

        # Tokenize and intersperse text once per sentence
        text_id_sequence = intersperse(
            text_to_sequence(text, norm=True),
            0,
        )

        for speaker_id, speaker_name in SPEAKERS.items():
            filename = f"spk{speaker_id}_sent{sent_id}_{category}.wav"
            file_path = OUTPUT_DIR / filename
            rel_file_path = f"outputs/tts_baseline_evaluation/{filename}"

            item_start_time = datetime.datetime.now(datetime.timezone.utc).isoformat()
            t0 = time.perf_counter()

            try:
                # Perform inference
                with torch.no_grad():
                    audio_data = kinya_tts(text_id_sequence, speaker_id, speed=SPEED)

                t_infer = time.perf_counter() - t0

                # Audio processing (Float tensor -> 16-bit Mono PCM)
                audio = audio_data.detach().cpu().numpy()
                audio = np.squeeze(audio)
                audio = np.clip(audio, -1.0, 1.0)
                pcm_audio = (audio * 32767).astype(np.int16)

                # Save WAV file
                with wave.open(str(file_path), "wb") as wav_file:
                    wav_file.setnchannels(1)  # Mono
                    wav_file.setsampwidth(2)  # 16-bit PCM (2 bytes)
                    wav_file.setframerate(SAMPLE_RATE)
                    wav_file.writeframes(pcm_audio.tobytes())

                # Programmatic verification
                wav_info = verify_wav_file(file_path, expected_sample_rate=SAMPLE_RATE)
                audio_dur = wav_info["duration_seconds"]
                rtf = t_infer / audio_dur if audio_dur > 0 else 0.0

                total_synthesis_time += t_infer
                total_audio_duration += audio_dur
                success_count += 1

                entry = {
                    "filename": filename,
                    "relative_path": rel_file_path,
                    "sentence_id": sent_id,
                    "category": category,
                    "speaker_id": speaker_id,
                    "speaker_name": speaker_name,
                    "input_text": text,
                    "description": description,
                    "synthesis_speed": SPEED,
                    "synthesis_latency_seconds": round(t_infer, 4),
                    "audio_duration_seconds": round(audio_dur, 4),
                    "real_time_factor": round(rtf, 4),
                    "sample_rate": SAMPLE_RATE,
                    "audio_format": "16-bit mono PCM WAV",
                    "channels": wav_info["nchannels"],
                    "bit_depth": wav_info["bit_depth"],
                    "frames": wav_info["nframes"],
                    "file_size_bytes": file_path.stat().st_size,
                    "generation_timestamp": item_start_time,
                    "status": "success",
                    "error": None,
                }
                manifest_entries.append(entry)

                print(
                    f"[{success_count + failure_count:02d}/{total_planned}] "
                    f"Spk {speaker_id} ({speaker_name}) | Sent {sent_id} ({category}) | "
                    f"Dur: {audio_dur:.2f}s | Latency: {t_infer:.2f}s | RTF: {rtf:.3f} | OK"
                )

            except Exception as e:
                t_infer = time.perf_counter() - t0
                failure_count += 1
                error_msg = str(e)
                print(
                    f"[{success_count + failure_count:02d}/{total_planned}] "
                    f"Spk {speaker_id} ({speaker_name}) | Sent {sent_id} | ERROR: {error_msg}"
                )

                entry = {
                    "filename": filename,
                    "relative_path": rel_file_path,
                    "sentence_id": sent_id,
                    "category": category,
                    "speaker_id": speaker_id,
                    "speaker_name": speaker_name,
                    "input_text": text,
                    "description": description,
                    "synthesis_speed": SPEED,
                    "synthesis_latency_seconds": round(t_infer, 4),
                    "audio_duration_seconds": 0.0,
                    "real_time_factor": None,
                    "sample_rate": SAMPLE_RATE,
                    "audio_format": "16-bit mono PCM WAV",
                    "channels": 1,
                    "bit_depth": 16,
                    "frames": 0,
                    "file_size_bytes": 0,
                    "generation_timestamp": item_start_time,
                    "status": "failure",
                    "error": error_msg,
                }
                manifest_entries.append(entry)

    run_end_iso = datetime.datetime.now(datetime.timezone.utc).isoformat()
    avg_rtf = total_synthesis_time / total_audio_duration if total_audio_duration > 0 else 0.0

    # 6. Construct and Save Manifest
    manifest_data = {
        "metadata": {
            "title": "MWIZERWA AI — KinyaFlex Baseline TTS Evaluation",
            "model_path": str(MODEL_PATH),
            "model_relative_path": "models/kinya-flex-tts/kinya_flex_tts_base_trained.pt",
            "model_sha256": checkpoint_sha256,
            "architecture": "MB-iSTFT-VITS2",
            "device": str(device),
            "sample_rate": SAMPLE_RATE,
            "audio_format": "16-bit mono PCM WAV",
            "synthesis_speed": SPEED,
            "total_sentences": len(BENCHMARK_SENTENCES),
            "total_speakers": len(SPEAKERS),
            "speakers": SPEAKERS,
            "total_expected_files": total_planned,
            "total_generated_files": success_count,
            "run_started_at": run_start_iso,
            "run_completed_at": run_end_iso,
        },
        "summary": {
            "total_audio_duration_seconds": round(total_audio_duration, 4),
            "total_synthesis_time_seconds": round(total_synthesis_time, 4),
            "average_real_time_factor": round(avg_rtf, 4),
            "success_count": success_count,
            "failure_count": failure_count,
            "all_passed": (failure_count == 0 and success_count == total_planned),
        },
        "entries": manifest_entries,
    }

    with open(MANIFEST_PATH, "w", encoding="utf-8") as f:
        json.dump(manifest_data, f, indent=2, ensure_ascii=False)

    print("-" * 70)
    print("Baseline Evaluation Complete!")
    print(f"Manifest saved to: {MANIFEST_PATH}")
    print(f"Total Successfully Generated: {success_count}/{total_planned}")
    print(f"Total Failures: {failure_count}")
    print(f"Total Audio Duration: {total_audio_duration:.2f} seconds ({total_audio_duration / 60:.2f} minutes)")
    print(f"Total Inference Time: {total_synthesis_time:.2f} seconds")
    print(f"Average Real-Time Factor (RTF): {avg_rtf:.3f}")
    print("=" * 70)


if __name__ == "__main__":
    main()
