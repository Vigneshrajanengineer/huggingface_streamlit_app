FROM python:3.9

WORKDIR /app
COPY . .
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

CMD ["streamlit", "run", "app/streamlit_app.py", "--server.port=8501", "--server.enableCORS=false"]
