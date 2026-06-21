# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Benchmark the efficiency of prefix caching.

This script allows you to benchmark the performance of
a model with and without prefix caching using either fixed prompts
or prompts sampled from the ShareGPT dataset.

Fixed example usage:
    python benchmark_prefix_caching.py \
        --model meta-llama/Llama-2-7b-chat-hf \
        --enable-prefix-caching \
        --num-prompts 1 \
        --repeat-count 100 \
        --input-length-range 128:256

Compare two models on the same prompts:
    python benchmark_prefix_caching.py \
        --model Qwen/Qwen2-7B-Instruct \
        --compare-model Qwen/Qwen3-8B \
        --enable-prefix-caching \
        --num-prompts 10 \
        --repeat-count 5 \
        --input-length-range 128:256

ShareGPT example usage:
    # This command samples 20 prompts with input lengths
    # between 128 and 256 tokens from the ShareGPT dataset,
    # then replicates each prompt 5 times.
    python benchmark_prefix_caching.py \
        --model meta-llama/Llama-2-7b-chat-hf \
        --dataset-path /path/to/ShareGPT_V3_unfiltered_cleaned_split.json \
        --enable-prefix-caching \
        --num-prompts 20 \
        --repeat-count 5 \
        --input-length-range 128:256
"""

import argparse
import dataclasses
import json
import random
import sys
import time
import traceback

from transformers import PreTrainedTokenizerBase

from vllm import LLM, RequestOutput, SamplingParams
from vllm.distributed import cleanup_dist_env_and_memory
from vllm.engine.arg_utils import EngineArgs
from vllm.utils.argparse_utils import FlexibleArgumentParser
from vllm.v1.metrics.reader import Counter, Metric

try:
    from vllm.tokenizers import get_tokenizer
except ImportError:
    from backend_request_func import get_tokenizer

PROMPT = "You are a helpful assistant in recognizes the content of tables in markdown format. Here is a table as fellows. You need to answer my question about the table.\n# Table\n|Opening|Opening|Sl. No.|Film|Cast|Director|Music Director|Notes|\n|----|----|----|----|----|----|----|----|\n|J A N|9|1|Agni Pushpam|Jayabharathi, Kamalahasan|Jeassy|M. K. Arjunan||\n|J A N|16|2|Priyamvada|Mohan Sharma, Lakshmi, KPAC Lalitha|K. S. Sethumadhavan|V. Dakshinamoorthy||\n|J A N|23|3|Yakshagaanam|Madhu, Sheela|Sheela|M. S. Viswanathan||\n|J A N|30|4|Paalkkadal|Sheela, Sharada|T. K. Prasad|A. T. Ummer||\n|F E B|5|5|Amma|Madhu, Srividya|M. Krishnan Nair|M. K. Arjunan||\n|F E B|13|6|Appooppan|Thikkurissi Sukumaran Nair, Kamal Haasan|P. Bhaskaran|M. S. Baburaj||\n|F E B|20|7|Srishti|Chowalloor Krishnankutty, Ravi Alummoodu|K. T. Muhammad|M. S. Baburaj||\n|F E B|20|8|Vanadevatha|Prem Nazir, Madhubala|Yusufali Kechery|G. Devarajan||\n|F E B|27|9|Samasya|Madhu, Kamalahaasan|K. Thankappan|Shyam||\n|F E B|27|10|Yudhabhoomi|K. P. Ummer, Vidhubala|Crossbelt Mani|R. K. Shekhar||\n|M A R|5|11|Seemantha Puthran|Prem Nazir, Jayabharathi|A. B. Raj|M. K. Arjunan||\n|M A R|12|12|Swapnadanam|Rani Chandra, Dr. Mohandas|K. G. George|Bhaskar Chandavarkar||\n|M A R|19|13|Thulavarsham|Prem Nazir, sreedevi, Sudheer|N. Sankaran Nair|V. Dakshinamoorthy||\n|M A R|20|14|Aruthu|Kaviyoor Ponnamma, Kamalahasan|Ravi|G. Devarajan||\n|M A R|26|15|Swimming Pool|Kamal Haasan, M. G. Soman|J. Sasikumar|M. K. Arjunan||\n\n# Question\nWhat' s the content in the (1,1) cells\n"  # noqa: E501

@dataclasses.dataclass
class Request:
    prompt: str
    prompt_len: int
    output_len: int

def _sum_counter_value(metrics: list[Metric], name: str) -> int:
    return sum(
        metric.value
        for metric in metrics
        if isinstance(metric, Counter) and metric.name == name
    )


def _prefix_cache_stats_from_outputs(
    outputs: list[RequestOutput],
) -> tuple[int, int]:
    queries = 0
    hits = 0
    for output in outputs:
        if output.prompt_token_ids is not None:
            queries += len(output.prompt_token_ids)
        if output.num_cached_tokens is not None:
            hits += output.num_cached_tokens
    return hits, queries


@dataclasses.dataclass
class GenerationSummary:
    num_requests: int
    expected_requests: int
    total_prompt_tokens: int
    total_output_tokens: int
    total_cached_tokens: int
    avg_prompt_tokens: float
    finish_reasons: dict[str, int]


@dataclasses.dataclass
class PrefixCacheHitRateResult:
    model: str
    hits: int
    queries: int
    hit_rate: float | None
    source: str
    load_s: float
    generate_s: float
    summary: GenerationSummary


def get_prefix_cache_hit_rate(
    llm: LLM, outputs: list[RequestOutput]
) -> tuple[int, int, float | None, str]:
    hits = 0
    queries = 0
    source = "engine metrics"

    try:
        metrics = llm.get_metrics()
        queries = _sum_counter_value(metrics, "vllm:prefix_cache_queries")
        hits = _sum_counter_value(metrics, "vllm:prefix_cache_hits")
    except AssertionError:
        source = "request outputs"

    if queries == 0:
        hits, queries = _prefix_cache_stats_from_outputs(outputs)
        source = "request outputs"

    if queries == 0:
        return 0, 0, None, source

    return hits, queries, hits / queries * 100, source


def _prompt_token_lengths(model: str, prompts: list[str]) -> list[int]:
    tokenizer = get_tokenizer(model, trust_remote_code=True)
    return [len(tokenizer.encode(prompt)) for prompt in prompts]


def _summarize_outputs(
    outputs: list[RequestOutput], expected_requests: int
) -> GenerationSummary:
    total_prompt_tokens = 0
    total_output_tokens = 0
    total_cached_tokens = 0
    finish_reasons: dict[str, int] = {}
    for output in outputs:
        if output.prompt_token_ids is not None:
            total_prompt_tokens += len(output.prompt_token_ids)
        if output.num_cached_tokens is not None:
            total_cached_tokens += output.num_cached_tokens
        if output.outputs:
            completion = output.outputs[0]
            total_output_tokens += len(completion.token_ids)
            reason = completion.finish_reason or "unknown"
            finish_reasons[reason] = finish_reasons.get(reason, 0) + 1
    num_requests = len(outputs)
    avg_prompt_tokens = (
        total_prompt_tokens / num_requests if num_requests > 0 else 0.0
    )
    return GenerationSummary(
        num_requests=num_requests,
        expected_requests=expected_requests,
        total_prompt_tokens=total_prompt_tokens,
        total_output_tokens=total_output_tokens,
        total_cached_tokens=total_cached_tokens,
        avg_prompt_tokens=avg_prompt_tokens,
        finish_reasons=finish_reasons,
    )


def _print_generation_summary(
    model: str,
    summary: GenerationSummary,
    *,
    expected_avg_prompt_tokens: float | None = None,
) -> None:
    print(
        f"Requests: {summary.num_requests}/{summary.expected_requests}, "
        f"prompt tokens: {summary.total_prompt_tokens} "
        f"(avg {summary.avg_prompt_tokens:.1f}), "
        f"output tokens: {summary.total_output_tokens}, "
        f"cached tokens: {summary.total_cached_tokens}",
        flush=True,
    )
    if summary.finish_reasons:
        print(f"Finish reasons: {summary.finish_reasons}", flush=True)

    if summary.num_requests != summary.expected_requests:
        print(
            f"WARNING: expected {summary.expected_requests} outputs but got "
            f"{summary.num_requests}. Results may be invalid.",
            flush=True,
        )
    if summary.total_output_tokens == 0:
        print(
            "WARNING: zero output tokens generated. "
            "The run likely did not perform real decoding.",
            flush=True,
        )
    if (
        expected_avg_prompt_tokens is not None
        and summary.avg_prompt_tokens < expected_avg_prompt_tokens * 0.5
    ):
        print(
            f"WARNING: avg prompt length for {model} is "
            f"{summary.avg_prompt_tokens:.1f} tokens, much shorter than the "
            f"{expected_avg_prompt_tokens:.1f} tokens sampled for --model. "
            "Cross-model text prompts often re-tokenize to very different "
            "lengths. Use --fair-compare to sample per model.",
            flush=True,
        )


def print_prefix_cache_hit_rate(llm: LLM, outputs: list[RequestOutput]) -> None:
    hits, queries, hit_rate, source = get_prefix_cache_hit_rate(llm, outputs)
    if hit_rate is None:
        print("Prefix cache hit rate: N/A (prefix caching disabled or no queries)")
        return

    print(
        f"Prefix cache hit rate: {hit_rate:.1f}% "
        f"({hits}/{queries} tokens, from {source})"
    )


def _destroy_llm(llm: LLM) -> None:
    print("Shutting down engine and releasing GPU memory...", flush=True)
    try:
        llm.llm_engine.engine_core.shutdown()
    except Exception as exc:
        print(f"Warning: engine shutdown failed: {exc}", flush=True)
    del llm
    cleanup_dist_env_and_memory()


def benchmark_model(
    model: str,
    args: argparse.Namespace,
    prompts: list[str],
    sampling_params: SamplingParams,
    *,
    expected_avg_prompt_tokens: float | None = None,
) -> PrefixCacheHitRateResult:
    engine_args = EngineArgs.from_cli_args(args)
    engine_args.model = model
    # LLM() defaults disable_log_stats=True; keep stats on for benchmark visibility.
    engine_args.disable_log_stats = False

    prompt_lens = _prompt_token_lengths(model, prompts)
    print(
        f"\n------ Model: {model} ------\n"
        f"Prompt token lengths for this model: "
        f"avg={sum(prompt_lens) / len(prompt_lens):.1f}, "
        f"min={min(prompt_lens)}, max={max(prompt_lens)}",
        flush=True,
    )

    print("Loading model (this may take a while)...", flush=True)
    load_start = time.time()
    llm = LLM.from_engine_args(engine_args)
    load_s = time.time() - load_start
    print(f"Model loaded in {load_s:.2f}s. Generating {len(prompts)} prompts...", flush=True)

    generate_start = time.time()
    outputs = llm.generate(prompts, sampling_params=sampling_params)
    generate_s = time.time() - generate_start

    summary = _summarize_outputs(outputs, expected_requests=len(prompts))
    _print_generation_summary(
        model,
        summary,
        expected_avg_prompt_tokens=expected_avg_prompt_tokens,
    )

    if llm.llm_engine.log_stats:
        # Force one stats log. Otherwise "Avg prompt throughput" only appears
        # every VLLM_LOG_STATS_INTERVAL seconds (default: 10s).
        llm.llm_engine.do_log_stats()

    hits, queries, hit_rate, source = get_prefix_cache_hit_rate(llm, outputs)
    print(
        f"Generate time {generate_s:.2f}s, load time {load_s:.2f}s, "
        f"total {load_s + generate_s:.2f}s",
        flush=True,
    )
    if hit_rate is None:
        print(
            "Prefix cache hit rate: N/A (prefix caching disabled or no queries)",
            flush=True,
        )
    else:
        print(
            f"Prefix cache hit rate: {hit_rate:.1f}% "
            f"({hits}/{queries} tokens, from {source})",
            flush=True,
        )

    _destroy_llm(llm)
    print(f"Finished benchmark for {model}.", flush=True)
    return PrefixCacheHitRateResult(
        model=model,
        hits=hits,
        queries=queries,
        hit_rate=hit_rate,
        source=source,
        load_s=load_s,
        generate_s=generate_s,
        summary=summary,
    )


def print_model_comparison(results: list[PrefixCacheHitRateResult]) -> None:
    print("\n====== Prefix cache hit rate comparison ======")
    for result in results:
        if result.hit_rate is None:
            hit_rate_str = "N/A"
        else:
            hit_rate_str = (
                f"{result.hit_rate:.1f}% "
                f"({result.hits}/{result.queries} tokens, from {result.source})"
            )
        print(
            f"{result.model}: {hit_rate_str}, "
            f"generate {result.generate_s:.2f}s, load {result.load_s:.2f}s, "
            f"prompt tokens {result.summary.total_prompt_tokens}, "
            f"output tokens {result.summary.total_output_tokens}"
        )

    valid_results = [result for result in results if result.hit_rate is not None]
    if len(valid_results) == 2:
        delta = valid_results[0].hit_rate - valid_results[1].hit_rate
        print(
            f"Difference: {valid_results[0].model} "
            f"{delta:+.1f}% vs {valid_results[1].model}"
        )
    print(
        "Note: unless --fair-compare is set, all models share the same text "
        "prompts but may tokenize them to very different lengths."
    )


def prepare_requests(args: argparse.Namespace, model: str) -> list[Request]:
    tokenizer = get_tokenizer(model, trust_remote_code=True)
    input_length_range = tuple(map(int, args.input_length_range.split(":")))
    if args.dataset_path is not None:
        if args.prefix_len > 0:
            raise ValueError(
                "prefix-len is not supported when dataset-path is provided."
            )
        print(f"Start to sample {args.num_prompts} prompts from {args.dataset_path}")
        return sample_requests_from_dataset(
            dataset_path=args.dataset_path,
            num_requests=args.num_prompts,
            tokenizer=tokenizer,
            input_length_range=input_length_range,
            fixed_output_len=args.output_len,
        )

    print(f"Start to sample {args.num_prompts} prompts from random")
    return sample_requests_from_random(
        num_requests=args.num_prompts,
        tokenizer=tokenizer,
        input_length_range=input_length_range,
        fixed_output_len=args.output_len,
        prefix_len=args.prefix_len,
    )


def prepare_prompts(
    args: argparse.Namespace, model: str | None = None
) -> tuple[list[str], float]:
    sample_model = model or args.model
    random.seed(args.seed)
    filtered_requests = prepare_requests(args, sample_model)

    # Print some helpful stats of the requests.
    print(f"Sampled {len(filtered_requests)} requests using {sample_model}.")
    prompt_lens = [req.prompt_len for req in filtered_requests]
    avg_prompt_len = sum(prompt_lens) / len(prompt_lens)
    print(f"Average input length: {avg_prompt_len}")
    print(f"P50 input length: {sorted(prompt_lens)[len(prompt_lens) // 2]}")
    print(f"Min Prompt Length: {min(prompt_lens)}")
    print(f"Max Prompt Length: {max(prompt_lens)}")

    print("Testing filtered requests")
    prompts = repeat_and_sort_requests(
        filtered_requests, repeat_count=args.repeat_count, sort=args.sort
    )
    return prompts, avg_prompt_len


def sample_tokens(tokenizer: PreTrainedTokenizerBase, length: int) -> list[int]:
    vocab = tokenizer.get_vocab()
    all_special_ids = set(tokenizer.all_special_ids)

    # Remove the special tokens.
    return random.choices(
        [v for v in vocab.values() if v not in all_special_ids],
        k=length,
    )


def sample_requests_from_dataset(
    dataset_path: str,
    num_requests: int,
    tokenizer: PreTrainedTokenizerBase,
    input_length_range: tuple[int, int],
    fixed_output_len: int | None,
) -> list[Request]:
    if fixed_output_len is not None and fixed_output_len < 4:
        raise ValueError("output_len too small")

    # Load the dataset.
    with open(dataset_path) as f:
        dataset = json.load(f)
    # Filter out the conversations with less than 2 turns.
    dataset = [data for data in dataset if len(data["conversations"]) >= 2]
    # Only keep the first two turns of each conversation.
    dataset = [
        (data["conversations"][0]["value"], data["conversations"][1]["value"])
        for data in dataset
    ]

    # Shuffle the dataset.
    random.shuffle(dataset)

    min_len, max_len = input_length_range
    assert min_len >= 0 and max_len >= min_len, "input_length_range too small"

    # Filter out sequences that are too long or too short
    filtered_requests: list[Request] = []

    for i in range(len(dataset)):
        if len(filtered_requests) == num_requests:
            break

        # Tokenize the prompts and completions.
        prompt_token_ids = tokenizer(dataset[i][0]).input_ids
        prompt = tokenizer.decode(prompt_token_ids)
        completion = dataset[i][1]
        completion_token_ids = tokenizer(completion).input_ids
        prompt_len = len(prompt_token_ids)
        output_len = (
            len(completion_token_ids) if fixed_output_len is None else fixed_output_len
        )
        if min_len <= prompt_len <= max_len:
            filtered_requests.append(Request(prompt, prompt_len, output_len))

    return filtered_requests


def sample_requests_from_random(
    num_requests: int,
    tokenizer: PreTrainedTokenizerBase,
    input_length_range: tuple[int, int],
    fixed_output_len: int | None,
    prefix_len: int,
) -> list[Request]:
    requests = []
    prefix_token_ids = sample_tokens(tokenizer, prefix_len)
    min_len, max_len = input_length_range

    for i in range(num_requests):
        unique_part_token_ids = sample_tokens(
            tokenizer, random.randint(min_len - prefix_len, max_len - prefix_len)
        )
        prompt_token_ids = prefix_token_ids + unique_part_token_ids
        prompt = tokenizer.decode(prompt_token_ids)
        prompt_len = len(prompt_token_ids)
        assert min_len <= prompt_len <= max_len, (
            f"prompt_len {prompt_len} out of range {min_len}:{max_len}"
        )
        requests.append(Request(prompt, prompt_len, fixed_output_len))
    return requests


def repeat_and_sort_requests(
    requests: list[Request], repeat_count: int, sort: bool = False
) -> list[str]:
    repeated_requests = requests * repeat_count
    if sort:
        repeated_requests.sort(key=lambda x: x[1])
    else:
        random.shuffle(repeated_requests)
    return [req.prompt for req in repeated_requests]


def main(args):
    if args.compare_model is not None and args.compare_model == args.model:
        raise ValueError("--compare-model must differ from --model")

    models = [args.model]
    if args.compare_model is not None:
        models.append(args.compare_model)

    sampling_params = SamplingParams(
        temperature=0,
        max_tokens=args.output_len,
        detokenize=not args.disable_detokenize,
    )

    shared_prompts: list[str] | None = None
    expected_avg_prompt_tokens: float | None = None
    if args.compare_model is None or not args.fair_compare:
        shared_prompts, expected_avg_prompt_tokens = prepare_prompts(args)

    print("------start generating------", flush=True)
    results: list[PrefixCacheHitRateResult] = []
    for model in models:
        try:
            if args.fair_compare and args.compare_model is not None:
                print(f"\nSampling prompts for {model} (--fair-compare)", flush=True)
                prompts, expected_avg = prepare_prompts(args, model=model)
                expected_avg_prompt_tokens = expected_avg
            else:
                assert shared_prompts is not None
                prompts = shared_prompts
            results.append(
                benchmark_model(
                    model,
                    args,
                    prompts,
                    sampling_params,
                    expected_avg_prompt_tokens=expected_avg_prompt_tokens,
                )
            )
        except Exception:
            print(
                f"ERROR: benchmark failed for model {model}",
                file=sys.stderr,
                flush=True,
            )
            traceback.print_exc()
            if len(models) == 1:
                raise

    if len(results) > 1:
        print_model_comparison(results)
    elif len(models) > 1 and len(results) < len(models):
        print(
            "\nComparison skipped: not all models completed successfully.",
            flush=True,
        )


def create_argument_parser():
    parser = FlexibleArgumentParser(
        description="Benchmark the performance with or without "
        "automatic prefix caching."
    )
    parser.add_argument(
        "--dataset-path", type=str, default=None, help="Path to the dataset."
    )
    parser.add_argument("--output-len", type=int, default=10)
    parser.add_argument(
        "--num-prompts",
        type=int,
        required=True,
        help="Number of the prompts sampled from dataset",
    )
    parser.add_argument(
        "--repeat-count",
        type=int,
        default=1,
        help="Number of times to repeat each prompt",
    )
    parser.add_argument(
        "--sort", action="store_true", help="Sort prompts by input length"
    )
    parser.add_argument(
        "--input-length-range",
        type=str,
        required=True,
        help="Range of input lengths for sampling prompts,"
        'specified as "min:max" (e.g., "128:256").',
    )
    parser.add_argument(
        "--prefix-len",
        type=int,
        default=0,
        help="Specifies the length of a common prefix to be "
        "added to the input prompt. The input-length-range will "
        "subtract this length when filtering prompts. Only used "
        "when dataset-path is not provided.",
    )
    parser.add_argument(
        "--compare-model",
        type=str,
        default=None,
        help=(
            "Optional second model to benchmark on the same prompts. "
            "Models are run sequentially and their prefix cache hit rates "
            "are compared at the end."
        ),
    )
    parser.add_argument(
        "--fair-compare",
        action="store_true",
        help=(
            "When comparing two models, sample prompts separately with each "
            "model's tokenizer so both sides target the same input-length-range. "
            "Without this flag, both models reuse the same text prompts, which "
            "often re-tokenize to very different lengths and makes runtime "
            "comparison misleading."
        ),
    )
    parser.add_argument(
        "--disable-detokenize",
        action="store_true",
        help=(
            "Do not detokenize responses (i.e. do not include "
            "detokenization time in the latency measurement)"
        ),
    )

    parser = EngineArgs.add_cli_args(parser)

    return parser


if __name__ == "__main__":
    parser = create_argument_parser()
    args = parser.parse_args()
    main(args)
