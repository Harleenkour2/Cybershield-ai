# CyberShield AI — Network Threat Defense Prototype

A multi-page Streamlit prototype for the P30 AI Cybersecurity Threat Defense Agent project.

## What works
- Multi-page dashboard and navigation
- Generated sample network-flow records
- CSV upload and validation
- Random Forest demo classification
- Severity prioritization and alert explanations
- Filters and record investigation
- Export analysis CSV and text incident report
- Demo model evaluation metrics and confusion matrix

## Important limitation
The built-in dataset and classifier are **synthetic teaching/demo data**. This is not a validated production intrusion-detection system and does not monitor live network traffic, block attacks, or establish that a real network is compromised. Do not present demo scores as real-world performance.

## Run on Windows
1. Install Python 3.10 or newer from python.org and tick **Add Python to PATH**.
2. Extract this ZIP.
3. Open the `cybershield_ai` folder in VS Code.
4. Open Terminal > New Terminal.
5. Run:

```bash
python -m venv .venv
.venv\Scripts\activate
python -m pip install -r requirements.txt
streamlit run app.py
```

6. Your browser should open the local website. If it does not, visit `http://localhost:8501`.

## CSV input schema
The current demo model requires these numeric columns:

- `flow_duration`
- `total_fwd_packets`
- `total_backward_packets`
- `flow_bytes_per_sec`
- `flow_packets_per_sec`
- `syn_flag_count`
- `dst_port`
- `packet_length_mean`

Optional display columns: `timestamp`, `source_ip`, `destination_ip`, `protocol`.

Real CICIDS2017/UNSW-NB15 files usually use different column names and schemas. A mapping/preprocessing step and careful dataset-specific evaluation are required before using them with this demo model.

## Suggested presentation wording
"CyberShield AI is a student prototype that demonstrates how network-flow features can be analyzed by a machine-learning classifier, then presented with severity prioritization and investigation summaries. The current built-in workflow uses synthetic data to demonstrate the interface; validating on a documented intrusion-detection dataset and adding authorized live-flow ingestion are future steps."

## Project structure
- `app.py`: interface, navigation, charts, user interactions
- `engine.py`: synthetic sample generator, model training, predictions, evaluation
- `requirements.txt`: dependencies
