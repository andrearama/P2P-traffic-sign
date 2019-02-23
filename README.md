# P2P-traffic-sign
Realistic traffic sign generator using Pix2Pix architecture.

Example video (each frame is processed indipendently) https://youtu.be/kExcKMxwg2c

## Requirements:
  - torch>=0.4.0
  - torchvision>=0.2.1
  - dominate>=2.3.1
  - visdom>=0.1.8.3
  - numpy
  - cv2
  - xml
  - PIL

## Run the network:
  - Install the listed requirements 

  - Clone the rempo:
  ```bash
  git clone https://github.com/andrearama/P2P-traffic-sign
  ```
  
  - Setup the environment:
  ```bash
  cd P2P-traffic-sign
   ./scripts/setup_environment.sh 
  ```
  - Put the generator architecture in under ./checkpoints/experiment_name/ 
  
  - Run the main file:
  ```bash
  python create_tested.py --dataroot ./datasets/traffic_signs --name sixth_trial --model pix2pix  --gpu_ids -1 --which_direction BtoA
  ```
