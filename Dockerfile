FROM python:3.9-slim

WORKDIR /app

COPY requirements.txt .

# SSL 에러 방지 및 라이브러리 설치 (소스코드는 복사하지 않음)
RUN pip3 install --no-cache-dir \
    --trusted-host pypi.python.org \
    --trusted-host pypi.org \
    --trusted-host files.pythonhosted.org \
    -r requirements.txt