#!/bin/bash
# Test AI trip worker endpoint

curl -X POST http://localhost:8091/v1/ai-trip/jobs \
  -H "Content-Type: application/json" \
  -H "X-Internal-Token: xxx" \
  -d '{
    "jobId": "test-123",
    "attemptId": "attempt-456",
    "request": {
      "tripName": "Test Trip",
      "cityName": "Hanoi"
    },
    "locale": "en",
    "callbackBaseUrl": "http://localhost:8080"
  }' | jq
