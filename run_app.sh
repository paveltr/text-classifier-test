sudo docker build -t text-classifier-test:latest .
sudo docker run --name text_classifier -d -p 5001:5001 text-classifier-test:latest