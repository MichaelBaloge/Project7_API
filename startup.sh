#!/bin/sh
apt update
apt-get install -y libgomp1
gunicorn --bind=0.0.0.0 --timeout 600 app:app
