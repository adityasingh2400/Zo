FROM nvidia/cuda:12.4.0-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y \
    python3 python3-pip python3-venv git wget ffmpeg \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# LiveTalking
RUN git clone https://github.com/lipku/LiveTalking.git /app/livetalking
WORKDIR /app/livetalking
RUN pip3 install -r requirements.txt || true

# TripoSR
RUN git clone https://github.com/VAST-AI-Research/TripoSR.git /app/triposr
WORKDIR /app/triposr
RUN pip3 install -r requirements.txt || true

WORKDIR /app

COPY start.sh /app/start.sh
RUN chmod +x /app/start.sh

EXPOSE 8010 8020

CMD ["/app/start.sh"]
