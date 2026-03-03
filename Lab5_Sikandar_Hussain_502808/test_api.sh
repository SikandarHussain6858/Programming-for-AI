#!/bin/bash

echo "API Testing Suite"
echo ""

BASE_URL="http://localhost:5000"

echo "1. Testing Health Endpoint..."
curl -s $BASE_URL/health
echo -e "\n"

echo "2. Testing Model Info Endpoint..."
curl -s $BASE_URL/model/info
echo -e "\n"

echo "3. Testing Single Prediction (First Class Female)..."
curl -X POST $BASE_URL/predict \
  -H "Content-Type: application/json" \
  -d '{
    "pclass": 1,
    "sex": "female",
    "age": 29.0,
    "sibsp": 0,
    "parch": 0,
    "fare": 211.3375,
    "embarked": "S",
    "deck": "B",
    "adult_male": false,
    "alone": true
  }'
echo -e "\n"

echo "4. Testing Single Prediction (Third Class Male)..."
curl -X POST $BASE_URL/predict \
  -H "Content-Type: application/json" \
  -d '{
    "pclass": 3,
    "sex": "male",
    "age": 22.0,
    "sibsp": 1,
    "parch": 0,
    "fare": 7.25,
    "embarked": "S",
    "adult_male": true,
    "alone": false
  }'
echo -e "\n"

echo "5. Testing Batch Prediction..."
curl -X POST $BASE_URL/predict \
  -H "Content-Type: application/json" \
  -d '[
    {
      "pclass": 1,
      "sex": "female",
      "age": 38.0,
      "sibsp": 1,
      "parch": 0,
      "fare": 71.2833,
      "embarked": "C",
      "adult_male": false,
      "alone": false
    },
    {
      "pclass": 3,
      "sex": "male",
      "age": 26.0,
      "sibsp": 0,
      "parch": 0,
      "fare": 7.925,
      "embarked": "S",
      "adult_male": true,
      "alone": true
    }
  ]'
echo -e "\n"

echo "6. Testing Error Handling (Missing Fields)..."
curl -X POST $BASE_URL/predict \
  -H "Content-Type: application/json" \
  -d '{"pclass": 1, "sex": "female"}'
echo -e "\n"

echo "7. Testing Error Handling (Invalid Input)..."
curl -X POST $BASE_URL/predict \
  -H "Content-Type: application/json" \
  -d '{"pclass": "invalid", "sex": "female", "age": "notanumber", "sibsp": 0, "parch": 0, "fare": 100, "embarked": "S"}'
echo -e "\n"

echo "All tests completed!"
