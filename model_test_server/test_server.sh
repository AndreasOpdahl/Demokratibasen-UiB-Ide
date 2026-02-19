#!/bin/bash
# Simple test script using curl (no Python dependencies needed)

SERVER_URL="${1:-http://localhost:8000}"

echo "Testing Model Summary Server at: $SERVER_URL"
echo ""

# Test health endpoint
echo "1. Testing /health endpoint..."
curl -s "$SERVER_URL/health" | python3 -m json.tool || echo "Failed to connect to server"
echo ""

# Test root endpoint
echo "2. Testing / endpoint..."
curl -s "$SERVER_URL/" | python3 -m json.tool || echo "Failed to connect to server"
echo ""

# Test summarize endpoint
echo "3. Testing /summarize endpoint..."
curl -s -X POST "$SERVER_URL/summarize" \
  -H "Content-Type: application/json" \
  -d '{
    "text": "Dette er en test tekst som skal oppsummeres. Den inneholder flere setninger for å teste om serveren fungerer korrekt.",
    "doc_type": "tekst",
    "max_length": 100,
    "min_length": 20
  }' | python3 -m json.tool || echo "Failed to connect to server or generate summary"
echo ""

echo "Test complete!"
