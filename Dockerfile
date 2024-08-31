FROM python:3.12
WORKDIR /app
COPY ./requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir --upgrade -r /app/requirements.txt
COPY ./dfpwm_tts /app/dfpwm_tts
CMD ["fastapi", "run", "dfpwm_tts/dfpwm_tts.py", "--port", "8000"]
