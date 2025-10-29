FROM python:3.12-slim
# dsa

WORKDIR /train

COPY . /train

RUN pip install -r requirements.txt

COPY . .

# ENV port=8080

# EXPOSE 8080

CMD ["python3","train.py"]