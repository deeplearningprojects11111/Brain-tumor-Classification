From python:3.9-sim as builder
RUN apt-get update &&
     apt-get install -y --no-install-recommends gcc python3-dev &&\
     rm-rf /var/lib/apt/lists

copy req.txt . 
RUN  pip install --user -r req.txt

FROM python:3.9-slim

WORKDIR /app

#copy neccasary file from builder stage

COPY  ---from=builder /root/.local /root/.local
COPY . .
#set environment variable
ENV PATH=/root/.local/bin:$PATH
ENV STREAMLIT_SERVER_PORT=8501
EXPOSE 8501
CMD ["streamlit", "run", "app.py" "--server.port=8051", "--server.address=0.0.0.0"]

