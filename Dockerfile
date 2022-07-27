from python:3.9

COPY . .
RUN pip install -e .[dev,notebooks]
