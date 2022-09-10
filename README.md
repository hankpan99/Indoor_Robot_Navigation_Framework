# Indoor_Robot_Navigation_Framework

## Introduction
Below is a demo video of navigating to the target `table` (highlighted with red mask) in `apartment_0` using `habitat-sim` simulator ([Link](https://github.com/facebookresearch/habitat-sim)).

![project](https://user-images.githubusercontent.com/75136798/189445287-5b079673-5740-45fa-8f2a-ed017ba6f412.gif)

## Reconstruction
In order to navigate to certain area, the agent needs to have knowledge about the environment first.

### Data Collection
Walking through the first floor (left figure) and second floor (right figure) in `apartment_0` meanwhile collecting `rgbd` and `semantic` images at each steps.

### Reconstruction
Unproject the depth image into 3D point clouds, then align point clouds to the first frame using `ICP algorithm`, we obtain the `3D rgb maps` of first floor (left figure) and second floor (right figure) in `apartment_0`.

![Screenshot](https://user-images.githubusercontent.com/75136798/189448499-a4c75350-20b6-4f6f-a725-e3f95ee4324a.png)

## Navigation
After creating maps for the environment, the agent can compute a collision-free trajectory and navigate to the target.

### Selecting Target
With `3D semantic map` (left figure), the agent knows about the location of each objects. In this example, we choose to navigate to the target `table` (middle figure).

### Path Planning
Using `RRT algorithm`, the agent can compute a collision-free trajectory to the target starting from current position (right figure).

### Navigation
The agent will perfrom `left`, `right`, `forward` these three actions according to the computed trajectory. The result can be seen on the top demo video.

![Screenshot](https://user-images.githubusercontent.com/75136798/189474623-b54fbca0-f98f-4f89-8b87-ab5afd8ec8f7.png)
