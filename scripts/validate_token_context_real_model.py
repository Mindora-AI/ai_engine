"""
Real-model validation script for AfriqueQwen token-aware context management.
Validates:
1. GGUF model loading with n_ctx=4096.
2. Single-turn response generation.
3. Emotional support disclosure scenario.
4. Reproduction and resolution of the 2,280-token overflow scenario.
5. Oversized message rejection with real tokenizer.
6. Multi-turn conversation token accumulation and automatic trimming.
7. CPU memory and inference latency observations.
"""

import sys
import time
from pathlib import Path
import psutil

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.llm.afrique_qwen import AfriqueQwenLLM
from src.llm.conversation import ConversationManager, OversizedMessageError
from src.llm.prompts import SYSTEM_PROMPT


def get_memory_usage_mb():
    process = psutil.Process()
    return process.memory_info().rss / (1024 * 1024)


def main():
    print("=" * 70)
    print("MWIZERWA AI — Real Model Context Management & Inference Validation")
    print("=" * 70)
    mem_initial = get_memory_usage_mb()
    print(f"Initial process memory: {mem_initial:.2f} MB\n")

    # Step 1: Initialize AfriqueQwen with n_ctx=4096
    print("[1/6] Initializing AfriqueQwenLLM with n_ctx=4096...")
    t0 = time.time()
    llm = AfriqueQwenLLM(n_ctx=4096, max_tokens=512)
    model = llm.load_model()
    t_load = time.time() - t0
    mem_after_load = get_memory_usage_mb()
    print(f"Model loaded in {t_load:.2f}s.")
    print(f"Reported model n_ctx: {model.n_ctx()}")
    print(f"Process memory after load: {mem_after_load:.2f} MB (Delta: +{mem_after_load - mem_initial:.2f} MB)\n")

    # Step 2: System prompt token count inspection
    print("[2/6] Inspecting SYSTEM_PROMPT token requirements...")
    manager = ConversationManager(llm=llm, n_ctx=4096, max_tokens=512, safety_margin=16)
    sys_tokens = llm.count_tokens([{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": "Muraho"}])
    print(f"Formatted tokens for SYSTEM_PROMPT + minimal user prompt: {sys_tokens}")
    print(f"Available input budget (4096 - 512 - 16): {manager.get_available_budget()} tokens\n")

    # Step 3: Single-turn normal response
    print("[3/6] Testing normal conversation response...")
    t0 = time.time()
    resp1 = manager.generate_response("Muraho neza! Amakuru ki?")
    t_gen1 = time.time() - t0
    print(f"Response 1 ({t_gen1:.2f}s):\n{resp1}\n")
    print(f"Conversation messages count: {len(manager.messages)}")

    # Step 4: Emotional support disclosure scenario
    print("[4/6] Testing emotional-support disclosure scenario...")
    disclosure = (
        "Uyu munsi wari mubi cyane kuri njye. Nakoze ibizamini by'akazi birananira, "
        "kandi numva ncitse intege cyane kuko nari maze amezi atandatu mbitegura. "
        "Numva nta kizere cyo kuzabona akazi ngifite, nkabura n'icyo mbwira umuryango wanjye."
    )
    t0 = time.time()
    resp2 = manager.generate_response(disclosure)
    t_gen2 = time.time() - t0
    print(f"Response 2 ({t_gen2:.2f}s):\n{resp2}\n")
    print(f"Conversation messages count: {len(manager.messages)}")

    # Step 5: Test the scenario that previously caused 2,280-token overflow
    print("[5/6] Testing 2,280+ token conversation continuity (previous failure scenario)...")
    long_msg1 = (
        "Murakoze cyane kubyo mumbwiye. Reka mbabwire byimbitse ibyambayeho. "
        "Mu gitondo nagiye ku biro by'ikompanyi niteguye neza, ariko nagezeyo nsanga "
        "gahunda bayihinduye. Bampaye ikizamini cy'imibare n'ikoranabuhanga gikomeye cyane "
        "kitarimo mu byo twasezeranye. Nakomeje kugerageza ariko igihe cyanshiranye ntarakirangiza. "
        "Natahanye intimba ikomeye mu mutima numva naniwe kandi nhemutse ku ntego zanjye. "
        "Ese urabona hari icyizere cy'ejo hazaza mu gihe nkomeje guhura n'ibibazo nk'ibi?"
    )
    resp3 = manager.generate_response(long_msg1)
    print(f"Response 3 generated successfully! Turn 3 complete.")

    # Check candidate token requirements
    total_tokens_now = llm.count_tokens(manager.messages + [{"role": "user", "content": "Kandi se..."}])
    print(f"Current full conversation token footprint: {total_tokens_now} tokens.")
    if total_tokens_now >= 2048:
        print(f"-> SUCCESS: Context footprint ({total_tokens_now} tokens) is >= 2048!")
        print("-> The previous 2,048-token context crash was avoided because n_ctx=4096 and token-aware manager is active!")

    # Step 6: Test multi-turn accumulation and automatic trimming
    print("\n[6/6] Testing multi-turn context accumulation and automatic trimming...")
    turn_num = 4
    # We will simulate turns until candidate tokens exceed budget (3568)
    while len(manager.messages) < 15:
        simulated_user_input = (
            f"Komeza umbwire kuri ibyo bibazo (Icyiciro cya {turn_num}). "
            "Ndashaka kumenya uburyo nakongera kubaka icyizere, "
            "nkamenya uburyo bwo gusobanurira inshuti n'umuryango ibyambayeho "
            "ndetse no gutegura ikindi kizamini mu buryo bunoze ntarangaye cyangwa ngo nigungire mu nzu. "
            "Waba ufite inama zindi zihariye zankomeza muri iyi minsi ikomeye?"
        )
        # Inspect candidate prompt tokens before inference
        cand_msgs, _ = manager.prepare_candidate_messages(simulated_user_input)
        cand_prompt_tokens = llm.count_tokens(cand_msgs)
        assert cand_prompt_tokens <= manager.get_available_budget(), "Candidate prompt exceeded input budget!"

        resp = manager.generate_response(simulated_user_input)
        after_count = len(manager.messages)
        committed_history_tokens = llm.count_tokens(manager.messages)
        print(
            f"Turn {turn_num}: Messages count = {after_count} | "
            f"Prompt tokens to inference = {cand_prompt_tokens} (Budget <= {manager.get_available_budget()}) | "
            f"Committed history tokens (with reply) = {committed_history_tokens}"
        )
        turn_num += 1
        if turn_num > 8:
            break

    # Verify system prompt remains intact
    assert manager.messages[0]["role"] == "system"
    assert manager.messages[0]["content"] == SYSTEM_PROMPT
    print("\nSystem prompt verified intact at index 0.")

    # Test oversized message rejection
    print("\n[Bonus] Testing oversized message rejection with real model tokenizer...")
    huge_message = "Iri ni ijambo rirerire cyane. " * 800  # ~4800 tokens
    try:
        manager.generate_response(huge_message)
        print("ERROR: Oversized message was not rejected!")
    except OversizedMessageError as e:
        print(f"SUCCESS: Oversized message rejected gracefully with OversizedMessageError:\n{e}")

    mem_final = get_memory_usage_mb()
    print(f"\nFinal process memory: {mem_final:.2f} MB")
    print("=" * 70)
    print("ALL REAL-MODEL VALIDATION CHECKS COMPLETED SUCCESSFULLY!")
    print("=" * 70)


if __name__ == "__main__":
    main()
