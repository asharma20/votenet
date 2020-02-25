# Setup Votenet
1. Clone repository
```bash
$ git clone https://github.com/asharma20/votenet.git
$ cd votenet
```
2. Create and activate virtualenv
```bash
$ python3 -m venv votenet_env
$ source votenet_env/bin/activate
```
3. Install dependencies
```bash
pip3 install -r requirements.txt
```
4. Compile CUDA layers for PointNet++
```bash
$ cd pointnet2
$ python3 setup.py install
```
5. Download pretrained models from README.md
6. Unzip pretrained models in root dir. This should generate a "demo_files" directory.
```bash
$ unzip votenet_pretrained_models.zip
```
7. Run example demo to test, expected output shown below
```bash
$ python3 demo.py
Constructed model.
Loaded checkpoint /home/anita/workspace/myvotenet/demo_files/pretrained_votenet_on_sunrgbd.tar (epoch: 180)
Loaded point cloud data: /home/anita/workspace/myvotenet/demo_files/input_pc_sunrgbd.ply
Inference time: 0.118590
Finished detection. 8 object detected.
Dumped detection results to folder /home/anita/workspace/myvotenet/demo_files/sunrgbd_results
```

# Run live demo with Kinect
1. Setup Kinect: follow directions in `Install Kinect related lib` of nod/nod/src/apps/nod_depth/how_to_run.md
2. Run inference on Kinect depth using pretrained sunrgbd model
```bash
$ python3 demo_live.py --visualize
```