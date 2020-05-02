FROM python:3.8

COPY requirements.txt /src/requirements.txt
RUN pip install -r /src/requirements.txt
WORKDIR /src
CMD [ "streamlit", "run", "simulation.py" ]
