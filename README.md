# PostureCoach 🤖

> **Currently in development** 👨‍💻

A real-time posture monitoring system that uses a YOLOv8 pose estimation model to classify user posture as "Good" or "Bad" from live camera input. The system is designed to provide immediate feedback to help users maintain proper sitting posture throughout their workday.

## Project Overview

PostureCoach combines computer vision and edge computing to create an intelligent posture monitoring solution:

- **Real-time Analysis**: Uses YOLOv8 pose estimation for instant posture classification
- **Edge Computing**: Deployed on Raspberry Pi 4 for low-latency inference
- **Custom Training**: Model trained on a carefully labeled dataset of sitting postures
- **User-Friendly Interface**: Frontend application for configuring Raspberry Pi settings
- **Immediate Feedback**: Provides real-time alerts for poor posture

## Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Camera Input  │───▶│  YOLOv8 Model   │───▶│  Posture Class  │
└─────────────────┘    └─────────────────┘    └─────────────────┘
                                │
                                ▼
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│  Frontend UI    │◀───│  Raspberry Pi   │◀───│  Real-time      │
│  (Settings)     │    │  (Edge Device)  │    │  Feedback       │
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

## Project Structure

```
PostureCoach/
├── client/          # Frontend application for Pi configuration
├── server/          # Backend services
├── pi/              # Raspberry Pi specific code
├── models/          # Trained YOLOv8 models
├── data/            # Training dataset and labels
├── training/        # Model training utilities
└── venv/           # Python virtual environment
```

## Features

- **Real-time Posture Detection**: Instant classification of sitting posture
- **Edge Inference**: Low-latency processing on Raspberry Pi 4
- **Custom Model**: Trained specifically for sitting posture scenarios
- **Configurable Settings**: Web interface for system configuration
- **Dataset Management**: Tools for cleaning and organizing training data

## Development Status

This project is currently in active development. Key components being worked on:

- [ ] Model training and optimization
- [ ] Raspberry Pi deployment pipeline
- [ ] Frontend configuration interface
- [ ] Real-time inference optimization
- [ ] User feedback system

## Dataset

The model is trained on a custom-labeled dataset of sitting postures. The dataset includes:

- **Source**: [Sitting Posture Dataset](https://universe.roboflow.com/ikornproject/sitting-posture-rofqf) from Roboflow
- **License**: CC BY 4.0
- **Provider**: Roboflow user (ikornproject)
- **Format**: YOLO format with 4 keypoint annotations
- **Classes**: Good posture vs Bad posture

### Dataset Structure

```
data/
├── train/          # Training images and labels
├── valid/          # Validation images and labels
├── test/           # Test images and labels
└── README.dataset.txt  # Dataset information
```

## Technology Stack

- **Computer Vision**: YOLOv8 (Ultralytics)
- **Edge Computing**: Raspberry Pi 4
- **Frontend**: Web-based configuration interface
- **Backend**: Python-based inference server
- **Data Format**: YOLO format with keypoint annotations

## Requirements

- Raspberry Pi 4 (4GB+ RAM recommended)
- USB Camera or Pi Camera Module
- Python 3.8+
- YOLOv8 dependencies
- Web browser for configuration interface

## Installation

_Installation instructions will be added as the project develops_

## License

This project is currently in development. The dataset used for training is licensed under CC BY 4.0.

## Contributing

This project is in active development. Contributions and feedback are welcome!

## Contact

For questions or contributions, please open an issue in the project repository.

---

**Note**: This project is currently in development. Features and documentation will be updated as the project progresses.

