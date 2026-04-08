FROM python:3.10

WORKDIR /code

COPY ./requirements.txt /code/requirements.txt
RUN pip install --no-cache-dir --upgrade -r /code/requirements.txt

COPY . /code

EXPOSE 7860
CMD ["sh", "-c", "python inference.py || true; python -m http.server 7860 --bind 0.0.0.0"]
