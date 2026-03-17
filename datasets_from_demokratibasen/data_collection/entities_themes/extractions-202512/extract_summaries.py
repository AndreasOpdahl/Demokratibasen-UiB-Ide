"""
Extract (JSON-structured) summaries from case documents.

This script is intentionally very similar to `extract_structured_data.py`,
but is designed to work with summarisation-style prompts such as
`summarisation-1-202602` (and similar), and it always appends the JSON schema
in **English** when requested via `Prompt.get_prompt(..., lang="en")`.
"""

import argparse
import json
import sys
import time
from pathlib import Path

import tiktoken

from llm_adapter import LLMAdapter, get_factory, detect_model_family
from llm_adapter.gpt_factory import estimate_tokens
from dataset_loader import DatasetLoader, list_datasets
from create_prompt import Prompt


# Control variables (defaults)
TEMPERATURE = 0.1
MAX_OUTPUT_TOKENS = 1000  # used by Demokratibasen since 2024-08-29
MAX_INPUT_TEXT_TOKENS = 2048  # Maximum tokens for document text (prompt tokens are separate)

OUTPUT_BASE_DIR = "extracted-data"

# Retry handling for malformed JSON outputs from providers
JSON_PARSE_RETRIES = 2
JSON_PARSE_RETRY_DELAY_SECONDS = 1.5
JSON_PARSE_RETRY_MAX_TOKENS_STEP = 500

# Bad response monitoring
BAD_RESPONSE_WINDOW = 10  # Number of documents to track in sliding window
MAX_BAD_RESPONSES = 3     # Maximum bad responses allowed in window
MAX_EXTRA_PROPERTIES_ACCEPTED = 3  # Maximum extra properties allowed in JSON response


class BadResponseMonitor:
    """
    Monitors bad responses in a sliding window.
    Tracks the last BAD_RESPONSE_WINDOW documents and counts bad responses.
    Only terminates if MAX_BAD_RESPONSES or more bad responses occur in the window.
    """

    def __init__(self, window_size: int = BAD_RESPONSE_WINDOW, max_bad: int = MAX_BAD_RESPONSES):
        self.window_size = window_size
        self.max_bad = max_bad
        self.responses: list[tuple[str, bool]] = []

    def record_response(self, doc_id: str, is_bad: bool):
        self.responses.append((doc_id, is_bad))
        if len(self.responses) > self.window_size:
            self.responses.pop(0)

    def should_terminate(self) -> tuple[bool, str]:
        if len(self.responses) < self.window_size:
            return False, ""
        bad_count = sum(1 for _, is_bad in self.responses if is_bad)
        if bad_count >= self.max_bad:
            bad_doc_ids = [doc_id for doc_id, is_bad in self.responses if is_bad]
            return True, (
                f"Terminating: {bad_count} bad responses in last {self.window_size} documents "
                f"(max allowed: {self.max_bad}). "
                f"Bad document IDs: {', '.join(bad_doc_ids[-bad_count:])}"
            )
        return False, ""


def _truncate_text_to_tokens(text: str, max_tokens: int, model_name: str, system_prompt_tokens: int = 0) -> str:
    """Truncate text to fit within max_tokens."""
    if max_tokens is None:
        return text

    available_tokens = max(0, max_tokens - system_prompt_tokens)
    if available_tokens <= 0:
        return ""

    try:
        enc = tiktoken.encoding_for_model(model_name)
    except KeyError:
        enc = tiktoken.get_encoding("cl100k_base")

    encoded = enc.encode(text)
    if len(encoded) <= available_tokens:
        return text

    truncated_encoded = encoded[:available_tokens]
    return enc.decode(truncated_encoded)


def _extract_token_usage(response) -> tuple[int, int]:
    """
    Extract token usage from response object.
    Supports OpenAI-style and some alternative formats.
    """
    input_tokens = 0
    output_tokens = 0

    if hasattr(response, "_response"):
        return _extract_token_usage(response._response)

    if hasattr(response, "usage"):
        usage = response.usage
        if hasattr(usage, "prompt_tokens"):
            input_tokens = usage.prompt_tokens or 0
        if hasattr(usage, "completion_tokens"):
            output_tokens = usage.completion_tokens or 0
        if input_tokens > 0 or output_tokens > 0:
            return input_tokens, output_tokens
        if hasattr(usage, "input_tokens"):
            input_tokens = usage.input_tokens or 0
        if hasattr(usage, "output_tokens"):
            output_tokens = usage.output_tokens or 0

    elif hasattr(response, "usage_metadata"):
        usage_metadata = response.usage_metadata
        if hasattr(usage_metadata, "prompt_token_count"):
            input_tokens = usage_metadata.prompt_token_count or 0
        if hasattr(usage_metadata, "candidates_token_count"):
            output_tokens = usage_metadata.candidates_token_count or 0

    return input_tokens, output_tokens


def _extract_json_from_response(response_text: str) -> dict:
    """
    Extract JSON from response text, handling markdown code blocks if present.
    Copied (lightly simplified) from `extract_structured_data.py`.
    """
    import re

    if response_text is None:
        raise ValueError("Could not extract valid JSON from response: response text is None")
    if not isinstance(response_text, str):
        raise ValueError(f"Could not extract valid JSON from response: expected string, got {type(response_text).__name__}")
    if not response_text.strip():
        raise ValueError("Could not extract valid JSON from response: response text is empty")

    MAX_TEXT_LENGTH = 100000
    texts_to_search: list[str] = []
    if len(response_text) > MAX_TEXT_LENGTH:
        texts_to_search.append(response_text[:MAX_TEXT_LENGTH])
        texts_to_search.append(response_text[-MAX_TEXT_LENGTH:])
    else:
        texts_to_search.append(response_text)

    try:
        return json.loads(response_text)
    except json.JSONDecodeError:
        pass

    json_patterns = [
        r"```json\s*\n(.*?)\n```",
        r"```\s*\n(.*?)\n```",
        r"```json\s*(.*?)```",
        r"```\s*(.*?)```",
    ]

    for text_to_search in texts_to_search:
        for pattern in json_patterns:
            match = re.search(pattern, text_to_search, re.DOTALL)
            if match:
                try:
                    return json.loads(match.group(1))
                except json.JSONDecodeError:
                    continue

    for text_to_search in texts_to_search:
        first_brace = text_to_search.find("{")
        if first_brace == -1:
            continue
        last_brace = text_to_search.rfind("}", first_brace)
        if last_brace == -1 or last_brace <= first_brace:
            continue
        potential_json = text_to_search[first_brace:last_brace + 1]
        try:
            return json.loads(potential_json)
        except json.JSONDecodeError:
            continue

    preview_length = min(200, len(response_text))
    preview = response_text[:preview_length]
    if len(response_text) > preview_length:
        preview += "..."
    raise ValueError(
        f"Could not extract valid JSON from response (length: {len(response_text)} chars). "
        f"Preview: {preview}"
    )


def parse_arguments():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Extract JSON-structured summaries from case documents using LLM"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="gpt-4o-mini",
        help="Model name (default: gpt-4o-mini)",
    )
    parser.add_argument(
        "--max-documents",
        type=int,
        default=None,
        help="Maximum number of documents to process (default: None, process all)",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        choices=list_datasets(),
        help="Dataset name to use (must be registered in dataset_loader.dataset_registry)",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        required=True,
        help="Prompt name to use (e.g. 'summarisation-1-202602', required)",
    )
    parser.add_argument(
        "--max-output-tokens",
        type=str,
        default=None,
        help=(
            "Maximum output tokens (completion tokens). Use 'all' or number >= 100000 for 'all-tokens', "
            "or a number like '4096' (default: 1000)"
        ),
    )
    parser.add_argument(
        "--max-input-text-tokens",
        type=int,
        default=None,
        help=(
            "Maximum tokens for document text (prompt tokens are separate). "
            "If set, documents will be truncated to fit this limit. "
            "If omitted, no explicit limit is applied here (the LLM adapter/model window still applies)."
        ),
    )
    parser.add_argument(
        "--ignore-below",
        type=int,
        default=None,
        help="Skip documents whose input token count (prompt+text) is below this value",
    )
    parser.add_argument(
        "--ignore-above",
        type=int,
        default=None,
        help="Skip documents whose input token count (prompt+text) is above this value",
    )
    parser.add_argument(
        "--ignore-bad-responses",
        action="store_true",
        help=(
            "If set, never abort due to too many bad responses in the recent window. "
            "Overrides the usual limit of max 3 bad responses in 10."
        ),
    )

    return parser.parse_args()


def main():
    args = parse_arguments()

    model = args.model

    try:
        model_family = detect_model_family(model)
    except ValueError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
    max_documents = args.max_documents
    dataset_name = args.dataset
    prompt_name = args.prompt
    ignore_bad_responses = args.ignore_bad_responses
    ignore_below = args.ignore_below
    ignore_above = args.ignore_above
    max_input_text_tokens = args.max_input_text_tokens

    if args.max_output_tokens is None:
        max_output_tokens = MAX_OUTPUT_TOKENS
    elif args.max_output_tokens.lower() == "all":
        max_output_tokens = None
    else:
        try:
            max_output_tokens = int(args.max_output_tokens)
            if max_output_tokens >= 100000:
                max_output_tokens = None
        except ValueError:
            print(
                f"Error: --max-output-tokens must be 'all' or a number, got '{args.max_output_tokens}'",
                file=sys.stderr,
            )
            sys.exit(1)

    try:
        factory = get_factory(model_family)
    except (ImportError, AttributeError) as e:
        print(f"Error: Could not load factory for model family '{model_family}' (detected from '{model}'): {e}", file=sys.stderr)
        sys.exit(1)
        return

    try:
        llm_adapter = LLMAdapter(factory, model)
    except Exception as e:
        print(f"Error initializing LLM: {e}", file=sys.stderr)
        sys.exit(1)
        return

    text_loader = DatasetLoader(dataset_name)
    prompt_creator = Prompt(prompt_name)

    # Use the newer naming convention similar to `extract_gpt_inferences.py`:
    # <dataset-name>-max-<max-input-tokens>-input-tokens-max-<max-output-tokens>-output-tokens-<prompt-name>
    if max_input_text_tokens is None:
        input_tokens_str = "all-input-tokens"
    else:
        input_tokens_str = f"max-{max_input_text_tokens}-input-tokens"

    if max_output_tokens is None or max_output_tokens >= 100000:
        output_tokens_str = "all-output-tokens"
    else:
        output_tokens_str = f"max-{max_output_tokens}-output-tokens"

    task_name = f"{dataset_name}-{input_tokens_str}-{output_tokens_str}-{prompt_name}"

    output_dir = Path(__file__).parent / OUTPUT_BASE_DIR / task_name / model
    output_dir.mkdir(parents=True, exist_ok=True)

    skipped_count = 0
    processed_count = 0
    attempt_count = 0

    total_input_tokens = 0
    total_output_tokens = 0

    bad_response_monitor = BadResponseMonitor()

    status_interval = 100
    last_status_time = time.time()
    last_status_input_tokens = 0
    last_status_output_tokens = 0
    start_time = time.time()

    max_output_tokens_display = (
        "all" if max_output_tokens is None or max_output_tokens >= 100000 else str(max_output_tokens)
    )
    print("=" * 80, file=sys.stderr)
    print("Starter prosessering (summarisation)", file=sys.stderr)
    print("-" * 80, file=sys.stderr)
    print(f"Modell: {model} (family: {model_family})", file=sys.stderr)
    print(f"Dataset: {dataset_name}", file=sys.stderr)
    print(f"Max output tokens: {max_output_tokens_display}", file=sys.stderr)
    print(f"Max input text tokens: {max_input_text_tokens}", file=sys.stderr)
    print(f"Prompt: {prompt_name}", file=sys.stderr)
    print(f"Output folder: {output_dir.relative_to(Path(__file__).parent)}", file=sys.stderr)
    print("=" * 80 + "\n", file=sys.stderr)

    for doc_id, kommune_nummer, kommune_navn, text in text_loader():
        if max_documents is not None and attempt_count >= max_documents:
            break

        output_file = output_dir / f"{doc_id}.json"
        if output_file.exists():
            skipped_count += 1
            continue

        should_skip = False
        skip_reason = None

        # For now, treat all model families the same in terms of JSON handling;
        # we always include the schema in English when requested.
        supports_only_json_object = model_family.lower() in ["mistral", "deepseek", "qwen"]

        try:
            prompt_preview = prompt_creator.get_prompt(
                kommune_navn,
                include_schema=supports_only_json_object,
                lang="en",
            )
            document_text_preview = prompt_creator.get_document_text(text)
            input_tokens_est = (
                estimate_tokens(prompt_preview, model_name=model)
                + estimate_tokens(document_text_preview, model_name=model)
            )
        except Exception:
            input_tokens_est = None

        if input_tokens_est is not None:
            if ignore_below is not None and input_tokens_est < ignore_below:
                should_skip = True
                skip_reason = (
                    f"input token count ({input_tokens_est}) < --ignore-below ({ignore_below})"
                )
            if ignore_above is not None and input_tokens_est > ignore_above:
                should_skip = True
                skip_reason = (
                    f"input token count ({input_tokens_est}) > --ignore-above ({ignore_above})"
                )

        if should_skip:
            skipped_count += 1
            if skip_reason:
                print(f"Skipping {doc_id}: {skip_reason}", file=sys.stderr)
            continue

        attempt_count += 1

        try:
            prompt = prompt_creator.get_prompt(
                kommune_navn,
                include_schema=supports_only_json_object,
                lang="en",
            )
            document_text = prompt_creator.get_document_text(text)

            document_text = _truncate_text_to_tokens(
                document_text,
                max_input_text_tokens,
                model_name=model,
                system_prompt_tokens=0,
            )

            actual_document_tokens_est = estimate_tokens(document_text, model_name=model)
            if max_input_text_tokens is not None and actual_document_tokens_est > max_input_text_tokens:
                print(f"ERROR: Document text token sanity check failed for {doc_id}", file=sys.stderr)
                print(
                    f"  Estimated document text tokens ({actual_document_tokens_est}) "
                    f"exceed max_input_text_tokens ({max_input_text_tokens})",
                    file=sys.stderr,
                )
                print("  Aborting immediately.", file=sys.stderr)
                sys.exit(1)

            extracted_data = None
            used_max_output_tokens = max_output_tokens
            for parse_attempt in range(JSON_PARSE_RETRIES + 1):
                retry_max_output_tokens = (
                    None
                    if max_output_tokens is None
                    else max_output_tokens + (parse_attempt * JSON_PARSE_RETRY_MAX_TOKENS_STEP)
                )
                used_max_output_tokens = retry_max_output_tokens
                response = llm_adapter.generate_text(
                    prompt=document_text,
                    system_prompt=prompt,
                    temperature=TEMPERATURE,
                    max_tokens=retry_max_output_tokens,
                    json_schema=prompt_creator.SCHEMA,
                )

                input_tokens, output_tokens = _extract_token_usage(response)

                if retry_max_output_tokens is not None and output_tokens > retry_max_output_tokens:
                    print(f"ERROR: Output token sanity check failed for {doc_id}", file=sys.stderr)
                    print(
                        f"  Actual output tokens ({output_tokens}) exceed max_output_tokens ({retry_max_output_tokens})",
                        file=sys.stderr,
                    )
                    print("  Aborting immediately.", file=sys.stderr)
                    sys.exit(1)

                total_input_tokens += input_tokens
                total_output_tokens += output_tokens

                if hasattr(response, "choices") and len(response.choices) > 0:
                    response_content = response.choices[0].message.content
                elif hasattr(response, "text"):
                    response_content = response.text
                else:
                    raise ValueError(f"Unexpected response format: {type(response)}")

                try:
                    extracted_data = _extract_json_from_response(response_content)
                    break
                except ValueError as parse_error:
                    if parse_attempt < JSON_PARSE_RETRIES:
                        next_max_tokens = (
                            "all"
                            if max_output_tokens is None
                            else str(max_output_tokens + ((parse_attempt + 1) * JSON_PARSE_RETRY_MAX_TOKENS_STEP))
                        )
                        print(
                            f"JSON parse failed for {doc_id} (attempt {parse_attempt + 1}/{JSON_PARSE_RETRIES + 1}): {parse_error}. "
                            f"Retrying in {JSON_PARSE_RETRY_DELAY_SECONDS:.1f}s with max_output_tokens={next_max_tokens}...",
                            file=sys.stderr,
                        )
                        time.sleep(JSON_PARSE_RETRY_DELAY_SECONDS)
                        continue
                    raise

            # For summarisation-style prompts we typically don't have a strict JSON Schema
            # in the JSON Schema sense; we therefore store the extracted object as-is.
            bad_response_monitor.record_response(doc_id, is_bad=False)

            output_record = {
                "dokument_id": doc_id,
                "kommune_nummer": kommune_nummer,
                "kommune_navn": kommune_navn,
                "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                "model": model,
                "temperature": TEMPERATURE,
                "max_output_tokens": used_max_output_tokens if used_max_output_tokens else "all",
                "max_input_text_tokens": max_input_text_tokens,
                "response": extracted_data,
            }

            with output_file.open("w", encoding="utf-8") as fout:
                json.dump(output_record, fout, ensure_ascii=False, indent=2)

            processed_count += 1
            print(f"Suksess: {doc_id}")

            if processed_count % status_interval == 0:
                current_time = time.time()
                time_since_last = current_time - last_status_time
                time_since_start = current_time - start_time

                input_tokens_since_last = total_input_tokens - last_status_input_tokens
                output_tokens_since_last = total_output_tokens - last_status_output_tokens

                avg_time_per_doc = time_since_last / status_interval
                avg_input_tokens_per_doc = input_tokens_since_last / status_interval
                avg_output_tokens_per_doc = output_tokens_since_last / status_interval

                print("\n" + "=" * 80, file=sys.stderr)
                print(f"Status etter {processed_count} vellykkede dokumenter ({skipped_count:,} hoppet over, allerede oppsummert)", file=sys.stderr)
                print("-" * 80, file=sys.stderr)
                print(f"Modell: {model}", file=sys.stderr)
                print(f"Dataset: {dataset_name}", file=sys.stderr)
                print(f"Max output tokens: {max_output_tokens_display}", file=sys.stderr)
                print(f"Max input text tokens: {max_input_text_tokens}", file=sys.stderr)
                print(f"Prompt: {prompt_name}", file=sys.stderr)
                print("-" * 80, file=sys.stderr)
                print(f"Tid siden forrige status: {time_since_last:.1f} sekunder", file=sys.stderr)
                print(f"Input tokens siden forrige: {input_tokens_since_last:,}", file=sys.stderr)
                print(f"Output tokens siden forrige: {output_tokens_since_last:,}", file=sys.stderr)
                print(f"Gjennomsnittlig tid per dokument: {avg_time_per_doc:.2f} sekunder", file=sys.stderr)
                print(f"Gjennomsnittlig input tokens per dokument: {avg_input_tokens_per_doc:.1f}", file=sys.stderr)
                print(f"Gjennomsnittlig output tokens per dokument: {avg_output_tokens_per_doc:.1f}", file=sys.stderr)
                print(f"Total tid: {time_since_start:.1f} sekunder", file=sys.stderr)
                print(f"Totale input tokens: {total_input_tokens:,}", file=sys.stderr)
                print(f"Totale output tokens: {total_output_tokens:,}", file=sys.stderr)
                print("=" * 80 + "\n", file=sys.stderr)

                last_status_time = current_time
                last_status_input_tokens = total_input_tokens
                last_status_output_tokens = total_output_tokens

        except Exception as e:
            error_str = str(e).lower()
            is_model_error = (
                ("404" in error_str and "model" in error_str)
                or ("model" in error_str and ("does not exist" in error_str or "not exist" in error_str or "not found" in error_str))
                or ("error code: 404" in error_str and "model" in error_str)
            )
            if is_model_error:
                print(f"Error: {e}", file=sys.stderr)
                print("Aborting: Model error will affect all documents.", file=sys.stderr)
                sys.exit(1)

            print(f"Hoppet over {doc_id} → {e}", file=sys.stderr)

            bad_response_monitor.record_response(doc_id, is_bad=True)
            should_terminate, terminate_msg = bad_response_monitor.should_terminate()
            if should_terminate and not ignore_bad_responses:
                print(f"Aborting: {terminate_msg}", file=sys.stderr)
                sys.exit(1)

            time.sleep(2)
            continue

    if skipped_count > 0:
        print(f"\nHoppet over {skipped_count} eksisterende dokumenter.", file=sys.stderr)

    # Save prompt and (informal) schema alongside results for inspection
    schema_file = output_dir / "prompt_schema.json"
    with schema_file.open("w", encoding="utf-8") as fout:
        json.dump(prompt_creator.SCHEMA, fout, ensure_ascii=False, indent=2)

    prompt_file = output_dir / "prompt.txt"
    with prompt_file.open("w", encoding="utf-8") as fout:
        fout.write(prompt_creator.PROMPT_TEMPLATE)

    print(
        f"\nFerdig. Prosesserte {processed_count} dokumenter (av {attempt_count} forsøk). "
        f"Data lagret i {output_dir.relative_to(Path.cwd())}"
    )

    print("\nToken-usage:")
    print(f"  INPUT tokens:  {total_input_tokens:,}")
    print(f"  OUTPUT tokens: {total_output_tokens:,}")
    print(f"  TOTAL tokens:  {total_input_tokens + total_output_tokens:,}")

    if processed_count > 0:
        mean_input_tokens = total_input_tokens / processed_count
        mean_output_tokens = total_output_tokens / processed_count
        print("\nPer dokument (gjennomsnitt):")
        print(f"  Dokumenter prosessert: {processed_count}")
        print(f"  INPUT tokens/dokument:  {mean_input_tokens:,.1f}")
        print(f"  OUTPUT tokens/dokument: {mean_output_tokens:,.1f}")
        print(f"  TOTAL tokens/dokument:  {(mean_input_tokens + mean_output_tokens):,.1f}")


if __name__ == "__main__":
    main()

