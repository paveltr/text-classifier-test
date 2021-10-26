curl -X 'POST' \
  'http://0.0.0.0:5001/predict_json/' \
  -H 'accept: application/json' \
  -H 'Content-Type: application/json' \
  -d @test_json/test_data.json