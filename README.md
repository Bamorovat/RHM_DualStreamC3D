# DualStreamC3D Model Training with RHM Dataset

This repository is focused on training the **DualStreamC3D** and **Multiview SlowFast** models using the **RHM (Robot House Multiview)** dataset. It is designed for advanced video analysis in Ambient Assistive Living Environments, utilizing multi-view video processing.

![RHM Dataset](RHM_sample_all.png)

## Files Description

- `config.py`: 
    - Configuration parameters for the **RHM** dataset.
    - Settings include dataset names, view types (e.g., OmniView, RobotView), and frame statuses (e.g., Normal, Subtract).

- `dataloader.py`: 
    - Data loading logic for the Two view grabbing the **RHM** dataset.
    - Defines `VideoDataset` class for handling different view types and frame statuses.

- `train.py`: 
    - Training pipeline for the DualStreamC3D and Multiview SlowFast models.
    - Integrates model definitions, data loading, and training loop.

- `models/DualStreamC3D`: 
    - Contains the DualStreamC3D model.

- `models/Multiview SlowFast`: 
    - Contains the Multiview SlowFast model.

- `splitlist/`:
    - Contains `testlist.txt`, `trainlist.txt`, and `vallist.txt`.
    - These files list the data splits for testing, training, and validation, respectively.


## System Requirements

- Python 3.x
- PyTorch
- OpenCV
- NumPy
- TensorBoardX
- Matplotlib
- Seaborn
- Pandas

## Installation

1. Clone the repository:
    ```bash
    git clone https://github.com/Bamorovat/RHM_DualStreamC3D.git
    ```

2. Install the required packages:
    ```bash
    pip install -r requirements.txt
    ```
   
3. Copy `testlist.txt`, `trainlist.txt`, and `vallist.txt` from the `splitlist/` folder to the root of the RHM dataset folder.


## Usage

1. Configure `config.py` for the RHM dataset.
2. Execute `train.py` to train the DualStreamC3D and Multiview SlowFast models:
    ```bash
    python train.py
    ```
3. `VideoDataset` in `dataloader.py` manages data handling.

## RHM Help

For assistance with obtaining the **RHM** dataset, send an email to Patrick at [p.holthaus@herts.ac.uk](mailto:p.holthaus@herts.ac.uk). More information about the Robot House, where the dataset was collected, can be found at [Robot House Website](https://robothouse.herts.ac.uk/).

## RHM Citation

If you are using the **RHM** dataset or this code in your research, please cite the following paper:

Bamorovat Abadi, M., Shahabian Alashti, M. R., Holthaus, P., Menon, C., & Amirabdollahian, F. (2023). RHM: Robot House Multi-View Human Activity Recognition Dataset. In ACHI 2023: The Sixteenth International Conference on Advances in Computer-Human Interactions. IARIA.

[Paper Link](https://www.thinkmind.org/index.php?view=article&articleid=achi_2023_4_160_20077)

Bibtex:
```
@inproceedings{bamorovat2023rhm,
title={Rhm: Robot house multi-view human activity recognition dataset},
author={Bamorovat Abadi, Mohammad and Shahabian Alashti, Mohamad Reza and Holthaus, Patrick and Menon, Catherine and Amirabdollahian, Farshid},
booktitle={ACHI 2023: The Sixteenth International Conference on Advances in Computer-Human Interactions},
year={2023},
organization={IARIA}
}
```


## License

This project is licensed under the GNU General Public License (GPL) v3.
