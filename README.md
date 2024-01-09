# Multi-View Video Analysis for Robotics

This repository contains code for a multi-view video analysis system, specifically designed for processing the RHM (Robotics Human Monitoring) dataset in robotics applications. The system is optimized for handling video data from multiple views, enhancing the capability of performing complex visual tasks in robotic environments.

## Files Description

- `config.py`: 
    - Configuration parameters tailored for the RHM dataset.
    - Settings include dataset names, view types (e.g., OmniView, RobotView), and frame statuses (e.g., Normal, Subtract).

- `dataloader.py`: 
    - Data loading logic for the RHM dataset.
    - Defines the `VideoDataset` class to handle various view types and frame statuses.

- `train.py`: 
    - Training pipeline for video analysis models on the RHM dataset.
    - Integrates model definitions, data loading, and the training process.
    - Models include `C3D_Multiview` and `SlowFast_Multiview`.

## System Requirements

- Python 3.x
- PyTorch (version specifics)
- OpenCV
- NumPy
- TensorBoardX
- Matplotlib
- Seaborn
- Pandas

## Installation

1. Clone the repository:
    ```bash
    git clone [repository-url]
    ```

2. Install the required packages:
    ```bash
    pip install -r requirements.txt
    ```

## Usage

1. Configure settings in `config.py` for the RHM dataset.
2. Execute `train.py` to start the training on the dataset:
    ```bash
    python train.py
    ```
3. The `VideoDataset` class from `dataloader.py` will be used for data handling.

## Contributing

Contributions to this project are welcome. Please adhere to the project's coding style and submit pull requests for any new features or bug fixes.

## License

This project is licensed under the [MIT License](LICENSE).

## Acknowledgments

- [Any collaborators or contributors]
- [Institutional affiliations, if any]
- [Funding sources, if any]
