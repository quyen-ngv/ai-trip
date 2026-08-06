#!/bin/bash
# Test DeepSeek API

API_KEY="sk-d3992cf35e364cc2af0da7e3e0ab187c"
BASE_URL="https://api.deepseek.com"

echo "=== Testing DeepSeek API ==="
echo ""

# Test 1: deepseek-chat model
echo "Test 1: deepseek-chat model"
curl -X POST "${BASE_URL}/chat/completions" \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer ${API_KEY}" \
  -d '{
    "model": "deepseek-chat",
    "messages": [
      {"role": "system", "content": "You are a helpful assistant."},
      {"role": "user", "content": "Hello"}
    ],
    "temperature": 0
  }'

echo ""
echo "---"
echo ""

# Test 2: deepseek-chat with JSON mode
echo "Test 2: deepseek-chat with JSON response_format"
curl -X POST "${BASE_URL}/chat/completions" \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer ${API_KEY}" \
  -d '{
    "model": "deepseek-chat",
    "messages": [
      {"role": "system", "content": "Return JSON only"},
      {"role": "user", "content": "Give me a JSON object with key test and value success"}
    ],
    "temperature": 0,
    "response_format": {"type": "json_object"}
  }'

echo ""
echo "---"
echo ""

# Test 3: List available models
echo "Test 3: List available models"
curl -X GET "${BASE_URL}/models" \
  -H "Authorization: Bearer ${API_KEY}"

echo ""
