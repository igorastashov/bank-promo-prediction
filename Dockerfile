FROM python:3.11-slim

EXPOSE 8501

RUN mkdir /streamlit-app

WORKDIR /streamlit-app

COPY . .

RUN pip install -r requirements.txt

CMD streamlit run app.py \
    --server.headless true \
    --browser.serverAddress="0.0.0.0" \
    --server.enableCORS false \
    --browser.gatherUsageStats false
