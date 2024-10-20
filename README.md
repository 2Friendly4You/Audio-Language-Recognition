# AI Language Identification Project

This project is an AI-based language identification system that can identify the language of spoken audio. The system uses a deep convolutional neural network audio classification model to classify audio data into different languages.

### Files and Directories

- **data/**: Directory containing audio data for different languages.
  - **de/**: German audio data.
  - **en/**: English audio data.
  - **es/**: Spanish audio data.
  - **fr/**: French audio data.
  - **it/**: Italian audio data.
  - **pt/**: Portuguese audio data.
  - **ru/**: Russian audio data.
  - **tr/**: Turkish audio data.
- **language_ai_api.py**: Python script for the Language AI API. NOT FINISHED
- **language_ai.html**: HTML file for the Language AI web interface. NOT FINISHED
- **language.ipynb**: Jupyter Notebook for experimenting with the Language AI model.
- **latest_model.keras**: The latest version of the trained model.
- **README.md**: This README file.

## Getting Started

### Prerequisites

- Python 3.12.7 recommended
- TensorFlow
- Jupyter Notebook

### Installation

1. Clone the repository
2. Install the required packages

### Usage

1. To train the model, run the Jupyter Notebook
2. To use the API, run the Python script
3. To view the web interface, open `language_ai.html` in your browser.

## Logs

Training and validation logs are stored in the `logs/fit/` directory, organized by training session timestamps.

## Contributing

Contributions are welcome! Please fork the repository and submit a pull request.
