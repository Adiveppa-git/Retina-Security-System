# Retina Security System

**AI/ML-Based Biometric Security with Anti-Spoofing and ESP32 Hardware Integration**

An integrated security project combining computer vision, biometric verification, anti-spoofing, a web application, and ESP32-based hardware integration to help detect and prevent unauthorized access.

## Project Overview

The Retina Security System brings together AI/ML models, a backend API, a React frontend, and embedded hardware firmware in one project.

The system is designed to evaluate biometric inputs, distinguish genuine inputs from spoof attempts, and support security alerts through connected hardware.

> **Note:** The exact biometric modality, supported camera inputs, and hardware alert workflow should be documented according to the final implementation.

## Key Components

### 1. AI/ML Models
- Anti-spoofing and liveness classification.
- Biometric detection and recognition modules.
- Model evaluation and prediction utilities.
- Python-based machine learning workflows.

### 2. Backend
- Python API and application services.
- User authentication and related endpoints.
- Data models and database integration.

### 3. Frontend
- React-based user interface.
- Registration, login, verification, and dashboard pages.
- API integration with the backend.

### 4. Hardware
- ESP32 firmware in `hardware/esp32_firmware/`.
- Arduino-compatible firmware source.
- Intended integration with the security system for hardware-based alerts.

## Technology Stack

| Component | Technologies |
|---|---|
| AI/ML | Python, PyTorch, OpenCV |
| Backend | Python, FastAPI |
| Frontend | React, JavaScript, CSS |
| Hardware | ESP32, Arduino framework |
| Data and tools | SQLite, Git, GitHub |

*Confirm the dependencies and actual hardware connections against your implementation before relying on this table.*

## Repository Structure

```text
Retina-Security-System/
├── ai_models/
│   ├── anti_spoofing/
│   ├── iris_detection/
│   ├── iris_recognition/
│   └── evaluate_all_models.py
├── backend/
├── dataset/
├── frontend/
│   └── react_app/
├── hardware/
│   └── esp32_firmware/
│       └── esp32_code.ino
├── project_images/
├── scratch/
├── requirements.txt
├── package.json
└── README.md
```

## Getting Started

### Prerequisites

Install the tools required for your project:

- Python compatible with the pinned dependencies.
- Node.js and npm.
- Git.
- Arduino IDE or PlatformIO for ESP32 development.
- Required model weights and authorized datasets.

### 1. Clone the Repository

```bash
git clone https://github.com/Adiveppa-git/Retina-Security-System.git
cd Retina-Security-System
```

### 2. Set Up the Python Environment

```bash
python -m venv .venv
```

Activate it on Windows:

```powershell
.\.venv\Scripts\Activate.ps1
```

Install dependencies:

```bash
python -m pip install -r requirements.txt
```

### 3. Set Up the Frontend

```bash
cd frontend/react_app
npm install
```

Start the frontend using the script configured in `frontend/react_app/package.json`. For example, if it defines a Vite development script:

```bash
npm run dev
```

### 4. Run the Backend

Return to the project root and follow the backend entry point and environment configuration in the source code. Configure required environment variables and database settings before starting the API.

### 5. Configure the ESP32

1. Open `hardware/esp32_firmware/esp32_code.ino` in Arduino IDE, or open the relevant firmware project in PlatformIO.
2. Select the correct ESP32 board and serial port.
3. Configure the pins and communication settings to match your actual circuit.
4. Connect the required hardware and upload the firmware.
5. Test the alert workflow before using the system in a real environment.

## Model Evaluation

The repository includes scripts for evaluating the AI/ML components.

Report measured results only after running the evaluation against a clearly identified test dataset. Include the dataset split, metrics, model checkpoint, and evaluation procedure when publishing accuracy or latency figures.

## Security and Privacy

- Never commit API keys, passwords, tokens, or private environment files.
- Do not publish biometric data or personal records without appropriate authorization.
- Keep private database files and sensitive test outputs out of the public repository.
- Validate authentication and hardware alert behavior before deployment.

## Limitations

Performance depends on the trained models, data quality, camera conditions, hardware configuration, and integration between components. This project should not be treated as a production-grade access-control system without further security and reliability testing.

## Future Improvements

- Improve and independently validate model performance.
- Add reproducible testing and evaluation reports.
- Strengthen authentication and access-control safeguards.
- Document the hardware circuit, pin mapping, and alert integration.
- Add setup instructions and screenshots for the complete workflow.

## Author

**Adiveppa Mamadapur**

GitHub: [@Adiveppa-git](https://github.com/Adiveppa-git)

---

*Developed as an AI/ML and embedded-systems security project.*