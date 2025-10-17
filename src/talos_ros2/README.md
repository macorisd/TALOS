# 3D Semantic Mapping with Voxeland and TALOS

This section provides the basic steps to set up a ROS 2 workspace with [Voxeland](https://github.com/MAPIRlab/Voxeland), [TALOS](https://github.com/macorisd/TALOS), and related repositories for generating 3D semantic maps from ScanNet scenes, using open-vocabulary semantic segmentation.

> **Note:** The following instructions assume your ROS 2 workspace is located at `/home/ubuntu/ros2_ws/`. If not, please adapt the paths accordingly. ROS 2 Humble is recommended.

## 1. Install and Configure Voxeland

Clone and configure Voxeland by following [its installation guide](https://github.com/MAPIRlab/Voxeland) in your `ros2_ws`.

## 2. Prepare Workspace Structure

After configuring Voxeland, your workspace (`ros2_ws`) should contain the following repositories and folders:

```
src/Voxeland
src/TALOS
src/instance_segmentation        # https://github.com/MAPIRlab/instance_segmentation
bag/ScanNet                      # https://github.com/josematez/ScanNet
```

* Ensure all repositories are initialized with their submodules and dependencies.

## 3. Build the Workspace

Clean previous build artifacts and execute `colcon build` to compile the workspace. Please adapt the paths if necessary:

```bash
cd ~/ros2_ws
rm -rf build/ install/ log/
colcon build --symlink-install --cmake-clean-cache
```

## 4. Launch the TALOS ROS 2 Node

Activate the TALOS Python environment, set the `PYTHONPATH`, source the ROS 2 setup and run the TALOS node. Please adapt the paths if necessary:

```bash
cd ~/ros2_ws
source /home/ubuntu/ros2_ws/src/TALOS/venvs/talos_env/bin/activate
export PYTHONPATH=$PYTHONPATH:/home/ubuntu/ros2_ws/src/TALOS/venvs/talos_env/lib/python3.10/site-packages
source install/setup.bash
ros2 run talos_ros2 talos_node
```

## 5. Run Voxeland and Play a ScanNet ROS Bag

Create and execute a bash script that contains the following commands:

```bash
cd ~/ros2_ws

# Init voxeland_robot_perception with TALOS detector
gnome-terminal -- bash -c "source ~/.bashrc; source /home/ubuntu/ros2_ws/venvs/voxenv/bin/activate; ros2 launch voxeland_robot_perception semantic_mapping.launch.py object_detector:=talos; exec bash"

# Init voxeland server
gnome-terminal -- bash -c "ros2 launch voxeland voxeland_server.launch.xml; exec bash"

# Open bag folder and play ros2 bag
gnome-terminal -- bash -c "cd /home/ubuntu/ros2_ws/bag/ScanNet/to_ros/ROS2_bags/scene0000_01/; ros2 bag play scene0000_01.db3; exec bash"
```

---

After completing these steps, RViz will open and display the semantic segmentation results from TALOS for the ScanNet scene images, while Voxeland incrementally builds the 3D semantic map.

For more details on TALOS or how to adjust runtime parameters, please refer to the [TALOS repository](https://github.com/macorisd/TALOS).
