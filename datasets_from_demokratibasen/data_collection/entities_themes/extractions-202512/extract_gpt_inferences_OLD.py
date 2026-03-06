"""
Extract GPT inferences (summaries, keywords, newsworthiness) from case documents.

Outputs JSON files with structure similar to raw_training_data CSVs:
- dok_id, kommune, url, dok_type, dok_tittel, text, model, max_output_tokens, max_input_text_tokens,
  oppsum_tittel, oppsummering, personer, nokkelord, nyhetsverdi

Usage:
    python extract_gpt_inferences.py --dataset dataset-202505 --max-output-tokens 1000 --max-input-tokens 128000 --prompt gpt-inferencing-202512 --model-family GPT
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import tiktoken

from llm_adapter import LLMAdapter, get_factory
from llm_adapter.gpt_factory import estimate_tokens
from dataset_loader.dataset_registry import get_dataset_path, get_dataset_adapter
from create_prompt import Prompt


# Control variables (defaults)
TEMPERATURE = 0.1
MAX_OUTPUT_TOKENS = 1000  # used by Demokratibasen since 2024-08-29
MAX_INPUT_TEXT_TOKENS = 2048  # Maximum tokens for document text (prompt tokens are separate)

OUTPUT_BASE_DIR = "extracted-data"

# Bad response monitoring
BAD_RESPONSE_WINDOW = 10
MAX_BAD_RESPONSES = 3
MAX_EXTRA_PROPERTIES_ACCEPTED = 3


class BadResponseMonitor:
    """Monitors bad responses in a sliding window."""
    
    def __init__(self, window_size: int = BAD_RESPONSE_WINDOW, max_bad: int = MAX_BAD_RESPONSES):
        self.window_size = window_size
        self.max_bad = max_bad
        self.responses = []
    
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
                f"(max allowed: {self.max_bad}). Bad document IDs: {', '.join(bad_doc_ids[-bad_count:])}"
            )
        return False, ""


def _truncate_text_to_tokens(text: str, max_tokens: int, model_name: str, system_prompt_tokens: int = 0) -> str:
    """
    Truncate text to fit within max_tokens.
    
    Args:
        text: Text to truncate
        max_tokens: Maximum tokens for the text (after accounting for system_prompt_tokens if provided)
        model_name: Model name for tiktoken encoding
        system_prompt_tokens: Number of tokens in system prompt (to reserve space, if needed)
    
    Returns:
        Truncated text
    """
    if max_tokens is None:
        return text
    
    # Calculate available tokens for the text (reserve space for prompt if provided)
    available_tokens = max(0, max_tokens - system_prompt_tokens)
    
    if available_tokens <= 0:
        return ""  # Not enough space even without text
    
    # Get encoding for the model
    try:
        enc = tiktoken.encoding_for_model(model_name)
    except KeyError:
        enc = tiktoken.get_encoding("cl100k_base")
    
    # Encode and truncate
    encoded = enc.encode(text)
    if len(encoded) <= available_tokens:
        return text
    
    truncated_encoded = encoded[:available_tokens]
    truncated_text = enc.decode(truncated_encoded)
    
    return truncated_text


def _extract_token_usage(response) -> tuple[int, int]:
    """Extract token usage from response object."""
    input_tokens = 0
    output_tokens = 0
    
    if hasattr(response, '_response'):
        return _extract_token_usage(response._response)
    
    if hasattr(response, 'usage'):
        usage = response.usage
        if hasattr(usage, 'prompt_tokens'):
            input_tokens = usage.prompt_tokens or 0
        if hasattr(usage, 'completion_tokens'):
            output_tokens = usage.completion_tokens or 0
        if input_tokens > 0 or output_tokens > 0:
            return (input_tokens, output_tokens)
        if hasattr(usage, 'input_tokens'):
            input_tokens = usage.input_tokens or 0
        if hasattr(usage, 'output_tokens'):
            output_tokens = usage.output_tokens or 0
    
    return (input_tokens, output_tokens)


def _extract_json_from_response(response_text: str) -> dict:
    """Extract JSON from response text, handling markdown code blocks if present."""
    import re
    
    if response_text is None:
        raise ValueError("Response text is None")
    if not isinstance(response_text, str):
        raise ValueError(f"Expected string, got {type(response_text).__name__}")
    if not response_text.strip():
        raise ValueError("Response text is empty")
    
    # Try to parse directly first
    try:
        return json.loads(response_text)
    except json.JSONDecodeError:
        pass
    
    # Try markdown code blocks
    json_patterns = [
        r'```json\s*\n(.*?)\n```',
        r'```\s*\n(.*?)\n```',
        r'```json\s*(.*?)```',
        r'```\s*(.*?)```',
    ]
    
    for pattern in json_patterns:
        match = re.search(pattern, response_text, re.DOTALL)
        if match:
            try:
                return json.loads(match.group(1))
            except json.JSONDecodeError:
                continue
    
    # Try to find JSON object boundaries
    first_brace = response_text.find('{')
    if first_brace != -1:
        last_brace = response_text.rfind('}', first_brace)
        if last_brace > first_brace:
            try:
                return json.loads(response_text[first_brace:last_brace + 1])
            except json.JSONDecodeError:
                pass
    
    raise ValueError(f"Could not extract valid JSON from response")


def _extract_json_schema(schema_data: dict) -> dict:
    """Extract the actual JSON schema from schema data structure."""
    if "schema" in schema_data:
        inner_schema = schema_data["schema"]
        if "parameters" in inner_schema:
            return inner_schema["parameters"]
        else:
            return inner_schema
    else:
        return schema_data


def _basic_schema_validation(json_data: dict, schema: dict) -> tuple[bool, str]:
    """Basic validation without jsonschema library."""
    if not isinstance(json_data, dict):
        return False, "Data is not a dictionary/object"
    
    required = schema.get("required", [])
    for field in required:
        if field not in json_data:
            return False, f"Missing required field: {field}"
    
    return True, ""


def parse_arguments():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Extract GPT inferences (summaries, keywords, newsworthiness) from case documents"
    )
    parser.add_argument(
        "--model-family",
        type=str,
        default="GPT",
        help="Model family (default: GPT)"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="gpt-4o-mini",
        help="Model name (default: gpt-4o-mini)"
    )
    parser.add_argument(
        "--max-documents",
        type=int,
        default=None,
        help="Maximum number of documents to process (default: None, process all)"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        choices=["dataset-202505", "dataset-202510", "dataset-Bergen-2017-2023"],
        help="Dataset name to use (required)"
    )
    parser.add_argument(
        "--prompt",
        type=str,
        required=True,
        help="Prompt name to use (required)"
    )
    parser.add_argument(
        "--max-output-tokens",
        type=str,
        default=None,
        help="Maximum output tokens (completion tokens). Use 'all' or number >= 100000 for 'all-tokens', or a number like '4096' (default: 1000)"
    )
    parser.add_argument(
        "--max-input-text-tokens",
        type=int,
        default=None,
        help=(
            "Maximum tokens for document text (prompt tokens are separate). "
            "If set, documents will be truncated to fit this limit. "
            "If omitted, no explicit limit is applied here (the LLM adapter/model window still applies)."
        )
    )
    parser.add_argument(
        "--ignore-below",
        type=int,
        default=None,
        help="Skip documents whose input token count (prompt+text) is below this value"
    )
    parser.add_argument(
        "--ignore-above",
        type=int,
        default=None,
        help="Skip documents whose input token count (prompt+text) is above this value"
    )
    parser.add_argument(
        "--ignore-bad-responses",
        action="store_true",
        help="If set, never abort due to too many bad responses"
    )
    
    return parser.parse_args()


def main():
    args = parse_arguments()
    
    model_family = args.model_family
    model = args.model
    max_documents = args.max_documents
    dataset_name = args.dataset
    prompt_name = args.prompt
    ignore_bad_responses = args.ignore_bad_responses
    ignore_below = args.ignore_below
    ignore_above = args.ignore_above
    # None means: no explicit limit here; rely on model/window size
    max_input_text_tokens = args.max_input_text_tokens
    if max_input_text_tokens is None:
        max_input_text_tokens_display = "none (model/window limit only)"
    else:
        max_input_text_tokens_display = str(max_input_text_tokens)
    
    # Parse max_output_tokens argument
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
            print(f"Error: --max-output-tokens must be 'all' or a number, got '{args.max_output_tokens}'", file=sys.stderr)
            sys.exit(1)

    
    # Get the factory
    try:
        factory = get_factory(model_family)
    except (ImportError, AttributeError) as e:
        print(f"Error: Invalid model family '{model_family}': {e}", file=sys.stderr)
        sys.exit(1)
    
    # Initialize components
    try:
        llm_adapter = LLMAdapter(factory, model)
    except Exception as e:
        print(f"Error initializing LLM: {e}", file=sys.stderr)
        sys.exit(1)
    
    # Load dataset adapter
    adapter_class = get_dataset_adapter(dataset_name)
    adapter = adapter_class()
    dataset_path = get_dataset_path(dataset_name)
    
    prompt_creator = Prompt(prompt_name)
    
    # Determine task name with new format:
    # <dataset-name>-max-<max-input-tokens>-input-tokens-max-<max-output-tokens>-output-tokens-<prompt-name>
    # Use "all" if no limit is specified
    
    if max_input_text_tokens is None:
        input_tokens_str = "all-input-tokens"
    else:
        input_tokens_str = f"max-{max_input_text_tokens}-input-tokens"
    
    if max_output_tokens is None or max_output_tokens >= 100000:
        output_tokens_str = "all-output-tokens"
    else:
        output_tokens_str = f"max-{max_output_tokens}-output-tokens"
    
    task_name = f"{dataset_name}-{input_tokens_str}-{output_tokens_str}-{prompt_name}"
    
    # Create output directory
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
    
    # Print configuration
    max_output_tokens_display = "all" if max_output_tokens is None or max_output_tokens >= 100000 else str(max_output_tokens)
    print("=" * 80, file=sys.stderr)
    print("Starter prosessering (GPT inferences)", file=sys.stderr)
    print("-" * 80, file=sys.stderr)
    print(f"Modell: {model}", file=sys.stderr)
    print(f"Dataset: {dataset_name}", file=sys.stderr)
    print(f"Max output tokens: {max_output_tokens_display}", file=sys.stderr)
    print(f"Max input text tokens: {max_input_text_tokens_display}", file=sys.stderr)
    print(f"Prompt: {prompt_name}", file=sys.stderr)
    print(f"Output folder: {output_dir.relative_to(Path(__file__).parent)}", file=sys.stderr)
    print("=" * 80 + "\n", file=sys.stderr)
    
    # Main loop - iterate through dataset (file or directory)
    def iter_raw_documents():
        """Iterator over raw documents from either JSONL file or directory of JSON files."""
        if dataset_path.is_file():
            # JSONL file: read line by line
            with open(dataset_path, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        yield json.loads(line)
                    except json.JSONDecodeError:
                        continue
        elif dataset_path.is_dir():
            # Directory: iterate through JSON files
            for json_file in sorted(dataset_path.glob("*.json")):
                try:
                    with open(json_file, 'r', encoding='utf-8') as f:
                        yield json.load(f)
                except (json.JSONDecodeError, Exception) as e:
                    print(f"Warning: Skipping {json_file.name}: {e}", file=sys.stderr)
                    continue
    
    for raw_doc in iter_raw_documents():
        if max_documents is not None and attempt_count >= max_documents:
            break
        
        # Check document type filter
        if hasattr(adapter, 'should_include_document'):
            if not adapter.should_include_document(raw_doc):
                continue
        
        # Normalize to get all fields
        normalized = adapter.normalize(raw_doc)
        
        doc_id = normalized["dok_id"]
        kommune_nummer = normalized["kommune_nummer"]
        kommune_navn = normalized["kommune_navn"]
        text = normalized["tekst"]
        url = normalized.get("url", "")
        dok_type = normalized.get("dok_type", "")
        dok_tittel = normalized.get("dok_tittel", "")
        
        # Filter empty/invalid text
        if not text or not text.strip() or len(text) < 10:
            continue
        
        # Determine output file path
        output_file = output_dir / f"{doc_id}.json"

        # If output already exists, always skip (already processed with this setting)
        if output_file.exists():
            skipped_count += 1
            continue

        # Check if we should skip based on input token count
        should_skip = False
        skip_reason = None
        
        # Estimate input token count for filtering (document text only)
        try:
            document_text_preview = prompt_creator.get_document_text(text)
            input_tokens_est = estimate_tokens(document_text_preview, model_name=model)
        except Exception as e:
            # If token estimation fails, proceed with processing (don't skip)
            input_tokens_est = None
        
        # Check ignore options if token estimation succeeded
        if input_tokens_est is not None:
            # Check ignore-below / ignore-above on *document text tokens only*
            if ignore_below is not None:
                if input_tokens_est < ignore_below:
                    should_skip = True
                    skip_reason = (
                        f"document text token count ({input_tokens_est}) "
                        f"< --ignore-below ({ignore_below})"
                    )
            
            # Check ignore-above (input tokens only)
            if ignore_above is not None:
                if input_tokens_est > ignore_above:
                    should_skip = True
                    skip_reason = (
                        f"document text token count ({input_tokens_est}) "
                        f"> --ignore-above ({ignore_above})"
                    )
        
        if should_skip:
            skipped_count += 1
            if skip_reason:
                print(f"Skipping {doc_id}: {skip_reason}", file=sys.stderr)
            continue

        attempt_count += 1

        try:
            prompt = prompt_creator.get_prompt(kommune_navn, include_schema=False)
            document_text = prompt_creator.get_document_text(text)
            
            # Optionally truncate document text to fit within max_input_text_tokens
            if max_input_text_tokens is not None:
                document_text = _truncate_text_to_tokens(
                    document_text, 
                    max_input_text_tokens, 
                    model_name=model,
                    system_prompt_tokens=0  # We're only limiting document text
                )
                
                # Sanity check: estimate actual document text tokens after truncation
                actual_document_tokens_est = estimate_tokens(document_text, model_name=model)
                
                if actual_document_tokens_est > max_input_text_tokens:
                    print(f"ERROR: Document text token sanity check failed for {doc_id}", file=sys.stderr)
                    print(f"  Estimated document text tokens ({actual_document_tokens_est}) exceed max_input_text_tokens ({max_input_text_tokens})", file=sys.stderr)
                    print(f"  Aborting immediately.", file=sys.stderr)
                    sys.exit(1)
            
            response = llm_adapter.generate_text(
                prompt=document_text,
                system_prompt=prompt,
                temperature=TEMPERATURE,
                max_tokens=max_output_tokens,
                json_schema=prompt_creator.SCHEMA
            )
            
            input_tokens, output_tokens = _extract_token_usage(response)
            
            # Sanity check: actual output tokens after API call
            if max_output_tokens is not None and output_tokens > max_output_tokens:
                print(f"ERROR: Output token sanity check failed for {doc_id}", file=sys.stderr)
                print(f"  Actual output tokens ({output_tokens}) exceed max_output_tokens ({max_output_tokens})", file=sys.stderr)
                print(f"  Aborting immediately.", file=sys.stderr)
                sys.exit(1)
            total_input_tokens += input_tokens
            total_output_tokens += output_tokens
            
            if hasattr(response, 'choices') and len(response.choices) > 0:
                response_content = response.choices[0].message.content
            elif hasattr(response, 'text'):
                response_content = response.text
            else:
                raise ValueError(f"Unexpected response format: {type(response)}")
            
            extracted_data = _extract_json_from_response(response_content)
            
            # Validate
            json_schema = _extract_json_schema(prompt_creator.SCHEMA)
            is_valid, error_message = _basic_schema_validation(extracted_data, json_schema)
            
            if not is_valid:
                bad_response_monitor.record_response(doc_id, is_bad=True)
                should_terminate, terminate_msg = bad_response_monitor.should_terminate()
                if should_terminate and not ignore_bad_responses:
                    print(f"Aborting: {terminate_msg}", file=sys.stderr)
                    sys.exit(1)
                raise ValueError(f"Schema validation failed: {error_message}")
            
            bad_response_monitor.record_response(doc_id, is_bad=False)
            
            # Create output record matching raw_training_data CSV structure
            output_record = {
                "dok_id": doc_id,
                "kommune": kommune_nummer,
                "url": url or "",
                "dok_type": dok_type or "",
                "dok_tittel": dok_tittel or "",
                "text": text,
                "model": model,
                "max_output_tokens": max_output_tokens if max_output_tokens else "all",
                "max_input_text_tokens": max_input_text_tokens,
                "oppsum_tittel": extracted_data.get("summary_title", ""),
                "oppsummering": extracted_data.get("summary_body", ""),
                "personer": ",".join(extracted_data.get("persons_mentioned", [])),
                "nokkelord": ",".join(extracted_data.get("keywords", [])),
                "nyhetsverdi": extracted_data.get("news_score", ""),
                # Also keep raw response for reference
                "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                "temperature": TEMPERATURE,
            }
            
            with output_file.open("w", encoding="utf-8") as fout:
                json.dump(output_record, fout, ensure_ascii=False, indent=2)
            
            processed_count += 1
            print(f"Suksess: {doc_id}")
            
            # Status output
            if processed_count % status_interval == 0:
                current_time = time.time()
                time_since_last = current_time - last_status_time
                input_tokens_since_last = total_input_tokens - last_status_input_tokens
                output_tokens_since_last = total_output_tokens - last_status_output_tokens
                
                max_output_tokens_display = "all" if max_output_tokens is None or max_output_tokens >= 100000 else str(max_output_tokens)
                print("\n" + "=" * 80, file=sys.stderr)
                print(f"Status etter {processed_count} vellykkede dokumenter", file=sys.stderr)
                print(f"Tid siden forrige status: {time_since_last:.1f} sekunder", file=sys.stderr)
                print(f"Tokens siden forrige: input={input_tokens_since_last:,}, output={output_tokens_since_last:,}", file=sys.stderr)
                print(f"Max output tokens: {max_output_tokens_display}", file=sys.stderr)
                print(f"Max input text tokens: {max_input_text_tokens}", file=sys.stderr)
                print("=" * 80 + "\n", file=sys.stderr)
                
                last_status_time = current_time
                last_status_input_tokens = total_input_tokens
                last_status_output_tokens = total_output_tokens
            
        except Exception as e:
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
    
    print(f"\nFerdig. Prosesserte {processed_count} dokumenter (av {attempt_count} forsøk). Data lagret i {output_dir}")
    print(f"\nToken-usage: INPUT={total_input_tokens:,}, OUTPUT={total_output_tokens:,}, TOTAL={total_input_tokens + total_output_tokens:,}")


if __name__ == "__main__":
    main()
