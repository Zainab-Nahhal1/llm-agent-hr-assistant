#!/usr/bin/env bash
# Example usage for the AI HR Assistant

curl -s -X POST http://127.0.0.1:5000/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "Generate interview questions for Data Scientist"}'

# Clear session
# curl -X POST http://127.0.0.1:5000/clear
