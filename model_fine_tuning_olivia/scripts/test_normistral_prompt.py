#!/usr/bin/env python3
"""
Test script to verify the correct prompt format for normistral-7b-instruct.
This will help us understand what format the model expects.
"""

from transformers import AutoTokenizer
import sys

def test_normistral_prompt_format():
    """Test different prompt formats to see which one works."""
    
    model_name = "norallm/normistral-7b-warm-instruct"
    
    print("=" * 70)
    print(f"Testing prompt format for: {model_name}")
    print("=" * 70)
    
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        print(f"✓ Tokenizer loaded successfully")
        
        # Check if tokenizer has chat_template
        has_chat_template = (
            hasattr(tokenizer, 'chat_template') and 
            tokenizer.chat_template is not None
        )
        
        print(f"\nTokenizer chat_template status: {has_chat_template}")
        
        if has_chat_template:
            print(f"\nChat template (first 500 chars):")
            print(tokenizer.chat_template[:500] if tokenizer.chat_template else "None")
        
        # Test input
        test_input = "Dette er en test tekst som skal oppsummeres."
        doc_type = "vedtak"
        
        print("\n" + "=" * 70)
        print("TESTING DIFFERENT PROMPT FORMATS")
        print("=" * 70)
        
        # Format 1: Using apply_chat_template (if available)
        if has_chat_template:
            print("\n1. Using tokenizer.apply_chat_template():")
            messages = [
                {
                    "role": "user",
                    "content": f"Du er en ekspert på tekstoppsummering. Oppsummer følgende {doc_type} på norsk:\n\n{test_input}"
                }
            ]
            try:
                formatted_1 = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                print(f"   Result (first 300 chars): {formatted_1[:300]}")
                print(f"   ✓ Success")
            except Exception as e:
                print(f"   ✗ Failed: {e}")
        else:
            print("\n1. Skipped (no chat_template available)")
        
        # Format 2: Manual format (what we're currently using)
        print("\n2. Using manual format (current approach):")
        manual_format = f"<s>[INST] Du er en ekspert på tekstoppsummering. Oppsummer følgende {doc_type} på norsk:\n\n{test_input} [/INST]"
        print(f"   Result (first 300 chars): {manual_format[:300]}")
        print(f"   ✓ Created")
        
        # Format 3: Check what special tokens are available
        print("\n3. Special tokens:")
        print(f"   bos_token: {tokenizer.bos_token} (id: {tokenizer.bos_token_id})")
        print(f"   eos_token: {tokenizer.eos_token} (id: {tokenizer.eos_token_id})")
        print(f"   pad_token: {tokenizer.pad_token} (id: {tokenizer.pad_token_id})")
        print(f"   unk_token: {tokenizer.unk_token} (id: {tokenizer.unk_token_id})")
        
        # Check for INST tokens
        if hasattr(tokenizer, 'vocab'):
            inst_tokens = [token for token in tokenizer.vocab.keys() if 'INST' in token or 'inst' in token]
            if inst_tokens:
                print(f"   INST-related tokens: {inst_tokens[:10]}")
        
        # Format 4: Tokenize both formats and compare
        print("\n4. Tokenization comparison:")
        if has_chat_template:
            try:
                messages = [
                    {
                        "role": "user",
                        "content": f"Du er en ekspert på tekstoppsummering. Oppsummer følgende {doc_type} på norsk:\n\n{test_input}"
                    }
                ]
                formatted_chat = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                tokens_chat = tokenizer.encode(formatted_chat, add_special_tokens=False)
                print(f"   Chat template tokens: {len(tokens_chat)} tokens")
                print(f"   First 20 token IDs: {tokens_chat[:20]}")
            except Exception as e:
                print(f"   Chat template tokenization failed: {e}")
        
        tokens_manual = tokenizer.encode(manual_format, add_special_tokens=False)
        print(f"   Manual format tokens: {len(tokens_manual)} tokens")
        print(f"   First 20 token IDs: {tokens_manual[:20]}")
        
        # Check if they're the same
        if has_chat_template:
            try:
                if formatted_chat == manual_format:
                    print(f"   ✓ Formats are identical")
                else:
                    print(f"   ⚠ Formats differ!")
                    print(f"   Difference at start: chat='{formatted_chat[:100]}' vs manual='{manual_format[:100]}'")
            except:
                pass
        
        print("\n" + "=" * 70)
        print("RECOMMENDATIONS")
        print("=" * 70)
        
        if has_chat_template:
            print("✓ Tokenizer has chat_template - USE apply_chat_template()")
            print("  This ensures consistency with the model's training format")
        else:
            print("⚠ Tokenizer lacks chat_template - using manual format")
            print("  Make sure the manual format matches what the model was trained on")
            print("  Consider setting chat_template manually (as we do in wandb_finetune.py)")
        
        print("\n" + "=" * 70)
        
    except Exception as e:
        print(f"✗ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    test_normistral_prompt_format()
