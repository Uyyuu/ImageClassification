FROM python:3.10.10

WORKDIR /frontend

COPY requirement.txt .

RUN pip install -r requirement.txt

# コピーしたら動いた
COPY . /frontend

ENTRYPOINT ["streamlit", "run"]

CMD ["app.py"]