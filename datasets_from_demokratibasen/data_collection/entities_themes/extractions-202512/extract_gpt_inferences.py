"""
Extract GPT inferences (summaries, keywords, newsworthiness) from case documents.

Outputs JSON files with structure similar to raw_training_data CSVs:
- dok_id, kommune, url, dok_type, dok_tittel, text, model, max_tokens,
  oppsum_tittel, oppsummering, personer, nokkelord, nyhetsverdi

Usage:
    python extract_gpt_inferences.py --dataset dataset-202505 --max-tokens all --prompt gpt-inferencing-202512 --model-family GPT
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

from llm_adapter import LLMAdapter, get_factory
from llm_adapter.gpt_factory import estimate_tokens
from dataset_loader.dataset_registry import get_dataset_path, get_dataset_adapter
from create_prompt import Prompt


# Control variables (defaults)
TEMPERATURE = 0.1
MAX_TOKENS = 2454  # = 2048 + 426, where 426 is the number of tokens in prompt+schema

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
        default="dataset-202505",
        choices=["dataset-202505", "dataset-202510"],
        help="Dataset name to use (default: dataset-202505)"
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default="gpt-inferencing-202512",
        help="Prompt name to use (default: gpt-inferencing-202512)"
    )
    parser.add_argument(
        "--max-tokens",
        type=str,
        default=None,
        help="Maximum tokens to send to the model. Use 'all' or number >= 100000 for 'all-tokens', or a number like '4096' (default: 4096)"
    )
    parser.add_argument(
        "--ignore-below-max",
        action="store_true",
        help=(
            "When used together with a numeric --max-tokens, skip documents whose "
            "prompt+text length (in tokens) is at or below max-tokens. "
            "Ignored if --max-tokens is 'all' or omitted."
        )
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
    
    # Parse max_tokens argument
    if args.max_tokens is None:
        max_tokens = MAX_TOKENS
    elif args.max_tokens.lower() == "all":
        max_tokens = None
    else:
        try:
            max_tokens = int(args.max_tokens)
            if max_tokens >= 100000:
                max_tokens = None
        except ValueError:
            print(f"Error: --max-tokens must be 'all' or a number, got '{args.max_tokens}'", file=sys.stderr)
            sys.exit(1)

    # Determine whether we should skip short documents when a numeric max-tokens is set
    # This is only meaningful when we have a concrete limit (not "all" / None)
    ignore_below_max = bool(args.ignore_below_max and (max_tokens is not None))
    
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
    
    # Load dataset with full metadata
    dataset_path = get_dataset_path(dataset_name)
    adapter_class = get_dataset_adapter(dataset_name)
    adapter = adapter_class()
    
    prompt_creator = Prompt(prompt_name)
    
    # Determine task name
    if max_tokens is None or max_tokens >= 100000:
        max_tokens_str = "all-tokens"
    else:
        max_tokens_str = f"{max_tokens}-tokens"
    
    task_name = f"{dataset_name}-{max_tokens_str}-{prompt_name}"
    
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
    max_tokens_display = "all" if max_tokens is None or max_tokens >= 100000 else str(max_tokens)
    print("=" * 80, file=sys.stderr)
    print("Starter prosessering (GPT inferences)", file=sys.stderr)
    print("-" * 80, file=sys.stderr)
    print(f"Modell: {model}", file=sys.stderr)
    print(f"Dataset: {dataset_name}", file=sys.stderr)
    print(f"Max tokens: {max_tokens_display}", file=sys.stderr)
    print(f"Prompt: {prompt_name}", file=sys.stderr)
    print(f"Output folder: {output_dir.relative_to(Path(__file__).parent)}", file=sys.stderr)
    print("=" * 80 + "\n", file=sys.stderr)
    
    # Main loop - iterate through dataset file directly to get full metadata
    with open(dataset_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            
            if max_documents is not None and attempt_count >= max_documents:
                break
            
            try:
                raw_doc = json.loads(line)
            except json.JSONDecodeError:
                continue
            
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

            # If requested, skip documents whose prompt+text length is at/below max_tokens
            if ignore_below_max and max_tokens is not None:
                try:
                    prompt_preview = prompt_creator.get_prompt(kommune_navn, include_schema=False)
                    document_text_preview = prompt_creator.get_document_text(text)
                    total_tokens_est = (
                        estimate_tokens(prompt_preview, model_name=model)
                        + estimate_tokens(document_text_preview, model_name=model)
                    )
                except Exception as e:
                    # If token estimation fails, fall back to processing the document
                    print(f"Warning: could not estimate tokens for {doc_id} ({e}), processing anyway", file=sys.stderr)
                    total_tokens_est = max_tokens + 1

                if total_tokens_est <= max_tokens:
                    # Considered "short enough" for this max-tokens run; skip
                    skipped_count += 1
                    continue

            attempt_count += 1

            try:
                prompt = prompt_creator.get_prompt(kommune_navn, include_schema=False)
                document_text = prompt_creator.get_document_text(text)
                
                response = llm_adapter.generate_text(
                    prompt=document_text,
                    system_prompt=prompt,
                    temperature=TEMPERATURE,
                    max_tokens=max_tokens,
                    json_schema=prompt_creator.SCHEMA
                )
                
                input_tokens, output_tokens = _extract_token_usage(response)
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
                    "max_tokens": max_tokens if max_tokens else "all",
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
                    
                    print("\n" + "=" * 80, file=sys.stderr)
                    print(f"Status etter {processed_count} vellykkede dokumenter", file=sys.stderr)
                    print(f"Tid siden forrige status: {time_since_last:.1f} sekunder", file=sys.stderr)
                    print(f"Tokens siden forrige: input={input_tokens_since_last:,}, output={output_tokens_since_last:,}", file=sys.stderr)
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
