# AI Log Monitor

**AI Log Monitor** is an intelligent, automated system designed to monitor, analyze, and identify irregularities in high-volume, high-velocity log data from distributed networks, cloud infrastructures, and workplace applications.

By integrating real-time analytics, data engineering, and machine learning, this system offers a robust pipeline that fetches logs from Elasticsearch, performs feature extraction (including NLP techniques like TF-IDF), and detects anomalies using the Isolation Forest algorithm.

## Features

- **Automated Log Ingestion**: Incrementally retrieves fresh logs from Elasticsearch (via Filebeat) to prevent redundancy.
- **Advanced Feature Engineering**: Extracts message lengths, time-based parameters, keyword flags, and textual features using TF-IDF vectorization.
- **Machine Learning Anomaly Detection**: Employs the Isolation Forest algorithm to detect deviations from normal behavior.
- **Severity Classification**: Categorizes anomalies into `CRITICAL`, `HIGH`, `MEDIUM`, `LOW`, and `INFO`.
- **Real-Time Alerting**: Sends prompt notifications for high-severity anomalies via a Telegram bot.
- **Analytics & Trend Analysis**: Uses KMeans clustering to identify recurring issues and tracks anomalies over time.
- **AI-Powered Assistant**: Leverages Google Gemini Large Language Models (LLMs) to interpret logs, providing structured explanations, root cause analyses, and suggested actions.
- **Visual Dashboards**: Integrates with Kibana for real-time visualization of metrics such as Anomaly Trends, Severity Distributions, and Hourly Heatmaps.
- **Automated Operations**: Built-in scheduling, model retraining, and health check validation.

## Prerequisites

- **OS**: Linux System (Tested on Xubuntu/Ubuntu)
- **ELK Stack**: Elasticsearch, Kibana, Logstash, Filebeat (v8.x)
- **Python**: Python 3.x
- **Google Gemini API Key**: For the AI Assistant features.
- **Telegram Bot Token & Chat ID**: For real-time alerting.

## Installation & Setup

### 1. ELK Stack Setup
Follow these steps to set up the Elastic stack (Elasticsearch, Kibana, Logstash, Filebeat):
1. **Add Elastic Repository:**
   ```bash
   wget -qO - https://artifacts.elastic.co/GPG-KEY-elasticsearch | sudo apt-key add -
   echo "deb https://artifacts.elastic.co/packages/8.x/apt stable main" | sudo tee /etc/apt/sources.list.d/elastic-8.x.list
   sudo apt update
   ```
2. **Install ELK:**
   ```bash
   sudo apt install -y elasticsearch kibana logstash filebeat
   ```
3. **Configure Elasticsearch Memory:**
   Set `-Xms512m` and `-Xmx512m` in `/etc/elasticsearch/jvm.options`.
4. **Enable & Start Services:**
   ```bash
   sudo systemctl daemon-reload
   sudo systemctl enable elasticsearch kibana logstash filebeat
   sudo systemctl start elasticsearch
   ```
5. **Generate Credentials & Tokens:**
   Generate a password for `elastic` user and an enrollment token for Kibana. Configure Kibana using `http://localhost:5601`.
6. **Filebeat Setup:**
   Enable system modules and configure `/etc/filebeat/filebeat.yml` to point to Elasticsearch and Kibana, then start the service.

### 2. System Dependencies
Set up the AI Log Monitor Python environment:
```bash
# Navigate to the project directory
# Create a virtual environment
python3 -m venv venv
source venv/bin/activate

# Install requirements
pip install -r requirements.txt
```

### 3. Environment Configuration
Create a `.env` file based on the `.env.example`:
```bash
cp .env.example .env
nano .env
```
Add your **Gemini API Key**, **Telegram Bot Token**, and **Elasticsearch** credentials into the `.env` file.

## Usage

### Health Check
Run the health check to verify connections to Elasticsearch, API availability, and environment setups:
```bash
python3 health_check.py
```

### Initial Pipeline Execution
Run the core pipeline scripts in sequence to fetch data, train the model, detect anomalies, and upload results:
```bash
python3 fetch_logs.py
python3 feature_engineering.py
python3 train_model.py
python3 predict.py
python3 upload.py
```

### Real-Time Monitoring
Open a terminal to start the real-time log watcher:
```bash
source venv/bin/activate
python3 realtime_watcher.py
```

### AI Assistant Chat
Interact with the AI assistant for contextual insights on recent logs:
```bash
python3 ai_assistant.py
```

### Test Log Generation
For testing the system's detection capabilities, generate simulated logs (including anomalies):
```bash
sudo python3 live_log_generator.py
```

## Kibana Dashboard
Create a comprehensive Kibana dashboard using the `ai-log-anomalies` index pattern to visualize:
- **Anomaly Trend (Line Chart)**
- **Severity Distribution (Pie Chart)**
- **Top Anomaly Logs (Data Table)**
- **Hourly Heatmap**
- **Critical Alerts Counter**

## Project Structure
- `fetch_logs.py`: Retrieves data from Elasticsearch.
- `feature_engineering.py`: Processes and creates NLP features using TF-IDF.
- `train_model.py`: Trains the Isolation Forest model for anomaly detection.
- `predict.py` / `detect_anomalies.py`: Detects anomalies in live data and assigns severity.
- `realtime_watcher.py`: Daemon for continuous anomaly detection.
- `ai_assistant.py` / `ai_assistant_llm.py`: Interfaces with the Gemini API to provide intelligent log insights.
- `alerts.py`: Handles notifications to Telegram.
- `health_check.py`: System readiness diagnostic tool.
- `trend_analysis.py` & `clustering.py`: Analytics for log trends and recurring issue grouping.

## Author
**Venkat Surya Kodati** (23STUCHH010355)  
*ICFAI Tech, Hyderabad*