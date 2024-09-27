
FROM python:3.9-slim


WORKDIR /app

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Expose the port the app runs on
EXPOSE 8000

# Set environment variables for OpenAI and Together API keys
ENV OPENAI_API_KEY=""
ENV TOGETHER_API_KEY=""
ENV ENABLED_ROUTERS=mf

# Define the command to run the API
CMD ["uvicorn", "routellm.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
