from vllm import LLM, SamplingParams
from vllm.distributed import cleanup_dist_env_and_memory
import time

if __name__ == "__main__":
    MODEL = "Qwen/Qwen3.5-9B"
    # or "tiny-random/qwen3-next-moe" for testing
    PROMPT_MULTIPLE = 310
    sampling_params = SamplingParams(temperature=0.0)
    prefix = (  # examples/offline_inference/prefix_caching.py
        "You are an expert school principal, skilled in effectively managing "
        "faculty and staff. Draft 10-15 questions for a potential first grade "
        "Head Teacher for my K-12, all-girls', independent school that emphasizes "
        "community, joyful discovery, and life-long learning. The candidate is "
        "coming in for a first-round panel interview for a 8th grade Math "
        "teaching role. They have 5 years of previous teaching experience "
        "as an assistant teacher at a co-ed, public school with experience "
        "in middle school math teaching. "
    )
    prefix2 = "Based on these information, fulfill " "the following paragraph: "
    prompt = PROMPT_MULTIPLE * prefix + prefix2 + "Hello, my name is"
    print("Prompt length:", len(prompt))
    for APC in [True, False]:
        engine = LLM(
            model=MODEL,
            enable_prefix_caching=APC,
            mamba_cache_mode="align",
            gpu_memory_utilization=0.9,
            disable_log_stats=False,
        )
        for i in range(3):
            if i == 0:
                print("Warm-up")
            if i == 1:
                print("Measuring")
                start_time = time.time()
            outputs = engine.generate(prompt, sampling_params)
            print("APC:", APC, i, f"Generated text: {outputs[0].outputs[0].text!r}")
            for m in engine.llm_engine.get_metrics():
                if "vllm:prefix_cache_hits" in m.name:
                    print(m.name, m.value)
        print("APC:", APC, "loop took --- %s seconds ---" % (time.time() - start_time))
        del engine
        cleanup_dist_env_and_memory()
