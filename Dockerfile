FROM python:3.9

WORKDIR /code
COPY ./requirements.txt /code/
RUN pip install --no-cache-dir --upgrade -v -r requirements.txt
RUN python -m spacy download en_core_web_sm

COPY ./app /code/app
COPY ./tests /code/tests


EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]



