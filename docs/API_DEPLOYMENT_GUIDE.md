# Egypt Tourism Assistant API Deployment Guide

## 🚀 Quick Start

### Local Development

1. **Install dependencies**:
   ```bash
   pip install fastapi uvicorn
   ```

2. **Start the API server**:
   ```bash
   python scripts/run_api_server.py
   ```

3. **Access the API**:
   - API: http://127.0.0.1:8000
   - Documentation: http://127.0.0.1:8000/docs
   - Alternative docs: http://127.0.0.1:8000/redoc

## 📋 API Endpoints

### Core Endpoints

| Endpoint | Method | Description | Example |
|----------|--------|-------------|---------|
| `/` | GET | Root endpoint with API info | `curl http://localhost:8000/` |
| `/predict` | POST | Get tourism advice | See examples below |
| `/health` | GET | Health check | `curl http://localhost:8000/health` |
| `/metrics` | GET | API usage metrics | `curl http://localhost:8000/metrics` |
| `/examples` | GET | Example questions | `curl http://localhost:8000/examples` |

### `/predict` Endpoint

**Request**:
```json
{
  "question": "What are the best tourist attractions in Egypt?",
  "max_length": 200,
  "temperature": 0.7
}
```

**Response**:
```json
{
  "question": "What are the best tourist attractions in Egypt?",
  "answer": "The top attractions in Egypt include the Great Pyramids of Giza...",
  "response_time": 2.34,
  "model_info": {
    "model_name": "Qwen/Qwen1.5-0.5B-Chat",
    "output_dir": "models/egypt_tourism_assistant",
    "fine_tuned": true
  }
}
```

## 🐳 Docker Deployment

### Build and Run

1. **Build the Docker image**:
   ```bash
   docker build -t egypt-tourism-api .
   ```

2. **Run the container**:
   ```bash
   docker run -p 8000:8000 egypt-tourism-api
   ```

### Docker Compose

1. **Start with Docker Compose**:
   ```bash
   docker-compose up --build
   ```

2. **Run in background**:
   ```bash
   docker-compose up -d
   ```

3. **View logs**:
   ```bash
   docker-compose logs -f egypt-tourism-api
   ```

4. **Stop the service**:
   ```bash
   docker-compose down
   ```

## 🧪 Testing the API

### Automated Testing

Run the test script:
```bash
python scripts/test_api.py
```

### Manual Testing with curl

1. **Health check**:
   ```bash
   curl http://localhost:8000/health
   ```

2. **Ask a question**:
   ```bash
   curl -X POST http://localhost:8000/predict \
     -H "Content-Type: application/json" \
     -d '{"question": "What currency is used in Egypt?"}'
   ```

3. **Get metrics**:
   ```bash
   curl http://localhost:8000/metrics
   ```

### Using Python requests

```python
import requests

# Ask a question
response = requests.post(
    "http://localhost:8000/predict",
    json={
        "question": "Do I need a visa to visit Egypt?",
        "max_length": 150,
        "temperature": 0.7
    }
)

print(response.json())
```

## 📊 Monitoring

### Health Checks

The API provides comprehensive health monitoring:

- **HTTP Health Check**: `GET /health`
- **Docker Health Check**: Built into the container
- **Metrics Collection**: `GET /metrics`

### Metrics Available

- Total requests
- Successful/failed requests
- Success rate percentage
- Average response time
- Model load status
- Server uptime

## ⚙️ Configuration

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `HOST` | `127.0.0.1` | Server host |
| `PORT` | `8000` | Server port |
| `RELOAD` | `false` | Auto-reload on code changes |

### Production Configuration

For production deployment:

1. **Set host to 0.0.0.0**:
   ```bash
   export HOST=0.0.0.0
   ```

2. **Use a reverse proxy** (nginx, traefik)
3. **Set up HTTPS** with SSL certificates
4. **Configure CORS** properly in `api_server.py`
5. **Set up monitoring** and logging
6. **Use a process manager** (systemd, supervisor)

## 🔒 Security Considerations

1. **CORS Configuration**: Update `allow_origins` in production
2. **Rate Limiting**: Consider adding rate limiting middleware
3. **Authentication**: Add API key authentication if needed
4. **Input Validation**: Already implemented with Pydantic
5. **HTTPS**: Use SSL/TLS in production

## 🚀 Production Deployment

### Using systemd (Linux)

1. **Create service file** `/etc/systemd/system/egypt-tourism-api.service`:
   ```ini
   [Unit]
   Description=Egypt Tourism Assistant API
   After=network.target

   [Service]
   Type=simple
   User=appuser
   WorkingDirectory=/path/to/your/app
   Environment=HOST=0.0.0.0
   Environment=PORT=8000
   ExecStart=/path/to/venv/bin/python scripts/run_api_server.py
   Restart=always

   [Install]
   WantedBy=multi-user.target
   ```

2. **Enable and start**:
   ```bash
   sudo systemctl enable egypt-tourism-api
   sudo systemctl start egypt-tourism-api
   ```

### Cloud Deployment

#### AWS EC2
1. Launch an EC2 instance
2. Install Docker
3. Clone repository
4. Run with Docker Compose

#### Google Cloud Run
1. Build container image
2. Push to Container Registry
3. Deploy to Cloud Run

#### Azure Container Instances
1. Build and push image
2. Create container instance
3. Configure networking

## 🔧 Troubleshooting

### Common Issues

1. **Model not loading**:
   - Check if fine-tuned model exists in `models/` directory
   - Verify model files are not corrupted

2. **Out of memory**:
   - Reduce `max_length` parameter
   - Use smaller batch sizes
   - Add swap memory

3. **Slow responses**:
   - Use GPU if available
   - Optimize model loading
   - Implement caching

4. **Port already in use**:
   ```bash
   export PORT=8001
   python scripts/run_api_server.py
   ```

### Logs and Debugging

- Check server logs for detailed error messages
- Use `/health` endpoint to verify model status
- Monitor `/metrics` for performance insights

## 📈 Performance Optimization

1. **Use GPU**: Deploy on GPU-enabled machines
2. **Model Quantization**: Already implemented with LoRA
3. **Caching**: Implement response caching for common questions
4. **Load Balancing**: Use multiple instances behind a load balancer
5. **Connection Pooling**: Optimize database connections if added

## 🔄 Updates and Maintenance

1. **Model Updates**: Replace model files and restart service
2. **Code Updates**: Use blue-green deployment
3. **Monitoring**: Set up alerts for health checks
4. **Backup**: Regularly backup model files and configurations 