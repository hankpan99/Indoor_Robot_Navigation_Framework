import os
import open3d as o3d
import numpy as np
import cv2
import quaternion
from scipy.spatial import distance

dataset_name = 'test_v3'
num_of_data = 152

'''data processing'''
# test_v3: 152, test_v4: 129, test_v5: 227
def data_processing(dataset = dataset_name, index = num_of_data+1):
  dep_list = []
  rgb_list = []
  sem_list = []

  # depth
  for i in range(1, index):
    tmp = cv2.imread(f'../../data/{dataset}/depth/step{i}.png', -1)
    dep_list.append(tmp)

  # rgb
  for i in range(1, index):
    tmp = cv2.imread(f'../../data/{dataset}/rgb/step{i}.png', 3)
    rgb_list.append(tmp)

  # semantic
  for i in range(1, index):
    tmp = cv2.imread(f'../../data/{dataset}/semantic/step{i}.png', 3)
    sem_list.append(tmp)
  
  return dep_list, rgb_list, sem_list

dep_list, rgb_list, sem_list = data_processing()
print(f'\nfinish data processing, data size = {len(dep_list)}')


'''generate point cloud'''
def depth_image_to_point_cloud(rgb, depth, K, depth_scale = 1000.0):
  v, u = np.mgrid[0:depth.shape[0], 0:depth.shape[1]]
  u = u.astype(float)
  v = v.astype(float)
  
  Z = depth.astype(float) / depth_scale
  X = (u - K[0, 2]) * Z / K[0, 0]  # (u-cx) * Z / fx
  Y = (v - K[1, 2]) * Z / K[1, 1]  # (v-cy) * Z / fy
  X = np.ravel(X)
  Y = np.ravel(Y)
  Z = np.ravel(Z)

  # remove points which is too far
  valid = Z < 0.08
  X = X[valid]
  Y = Y[valid]
  Z = Z[valid]

  position = np.vstack((X, Y, Z))

  R = np.ravel(rgb[:, :, 0])[valid]/255.
  G = np.ravel(rgb[:, :, 1])[valid]/255.
  B = np.ravel(rgb[:, :, 2])[valid]/255.
  
  points = np.transpose(position)
  colors = np.transpose(np.vstack((R, G, B)))

  return (points, colors)

def make_point_cloud_list(dep_list, rgb_list):
  tmp_point_cloud = []

  for i in range(len(dep_list)):
    # instrinct matrix
    K = np.array([[256, 0., 255],
                  [0., 255, 255],
                  [0., 0., 1]])

    # depth images to point clouds
    tmp_points, tmp_colors = depth_image_to_point_cloud(rgb_list[i], dep_list[i], K)

    # open3d declare
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(tmp_points[:, 0:3])
    pcd.colors = o3d.utility.Vector3dVector(tmp_colors[:, 0:3])

    tmp_point_cloud.append(pcd)
  
  return tmp_point_cloud

open3d_list = make_point_cloud_list(dep_list, rgb_list)
print('\nfinish generating point clouds')


"""global registration, compute icp, align point clouds"""
def preprocess_point_cloud(pcd, voxel_size):
    pcd_down = pcd.voxel_down_sample(voxel_size)

    radius_normal = voxel_size * 2
    pcd_down.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=30))

    radius_feature = voxel_size * 5
    pcd_fpfh = o3d.pipelines.registration.compute_fpfh_feature(
        pcd_down,
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=100))
    return pcd_down, pcd_fpfh

def prepare_dataset(voxel_size, id):
    source = open3d_list[id+1]
    target = open3d_list[id]

    source_down, source_fpfh = preprocess_point_cloud(source, voxel_size)
    target_down, target_fpfh = preprocess_point_cloud(target, voxel_size)

    return source, target, source_down, target_down, source_fpfh, target_fpfh

def execute_fast_global_registration(source_down, target_down, source_fpfh, target_fpfh, voxel_size):
    distance_threshold = voxel_size * 0.5
    result = o3d.pipelines.registration.registration_fast_based_on_feature_matching(
        source_down, target_down, source_fpfh, target_fpfh,
        o3d.pipelines.registration.FastGlobalRegistrationOption(
            maximum_correspondence_distance=distance_threshold))
    return result

def compute_icp_and_align_point_cloud(open3d_list, voxel_size = 0.002, threshold = 0.002):
  matrix = []
  
  for i in range(len(open3d_list)-1):
    source, target, source_down, target_down, source_fpfh, target_fpfh = prepare_dataset(voxel_size, i)
    result_fast = execute_fast_global_registration(source_down, target_down, source_fpfh, target_fpfh, voxel_size)

    reg_p2p = o3d.pipelines.registration.registration_icp(source_down, target_down, threshold, result_fast.transformation,
                                o3d.pipelines.registration.TransformationEstimationPointToPlane())
    matrix.append(reg_p2p.transformation)

  for i in range(1, len(matrix)):
    matrix[i] = np.dot(matrix[i-1], matrix[i])

  for i in range(1, len(open3d_list)):
    open3d_list[i].transform(matrix[i-1])
  
  return open3d_list, matrix

open3d_list, icp_matrix =  compute_icp_and_align_point_cloud(open3d_list)
print('\nfinish aligning point clouds')


def combined_and_down_sample(point_cloud_list, voxel_size):
  pcd_combined = o3d.geometry.PointCloud()
   
  for i in range(len(point_cloud_list)):
    pcd_combined += point_cloud_list[i]
  
  pcd_combined_down = pcd_combined.voxel_down_sample(voxel_size)
  pcd_combined_down.transform([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])

  return pcd_combined_down
  
pcd_combined_down = combined_and_down_sample(open3d_list, 0.0003)
print('\nfinish down sampling')


"""Adding trajectory"""
def test_ICP(icp_matrix):
  icp_traj = [[0, 0, 0]]

  for i in icp_matrix:  #[1:]:
    temp = [i[0][3], i[1][3], i[2][3]]
    icp_traj.append(temp)

  edges_traj = []
  
  for i in range(len(icp_traj)-1):
    edges_traj.append([i, i+1])
  colors = [[0, 0, 1] for i in range(len(edges_traj))]
  icp_line_set = o3d.geometry.LineSet(points = o3d.utility.Vector3dVector(icp_traj),
                                      lines = o3d.utility.Vector2iVector(edges_traj))
  icp_line_set.colors = o3d.utility.Vector3dVector(colors)
  icp_line_set.transform([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])

  return icp_line_set

def test_GT(icp_matrix, pcd_combined_down, dataset = dataset_name):
  file = open(f'../../data/{dataset}/pose.txt', 'r')
  icp_line_set = test_ICP(icp_matrix)
  points = []

  for cnt, line in enumerate(file):
    temp = line.split(' ')
    q = [float(temp[3]), float(temp[4]), float(temp[5]), float(temp[6])]
    t = [float(temp[0])/10.*255./1000., (float(temp[1]))/10.*255./1000., float(temp[2])/10.*255./1000.]
    q = quaternion.from_float_array(q)
    
    points.append(t[0:3])

    if cnt == 0:
      rotation = quaternion.as_rotation_matrix(q)
      
      camera_to_world = np.eye(4)
      camera_to_world[0:3, 0:3] = rotation
      camera_to_world[0:3, 3] = t

      pcd_combined_down.transform(camera_to_world)
      icp_line_set.transform(camera_to_world)

  edges = []
  for i in range(len(points)-1):
    edges.append([i, i+1])

  colors = [[1, 0, 0] for i in range(len(edges))]
  line_set = o3d.geometry.LineSet(points = o3d.utility.Vector3dVector(points),
                                  lines = o3d.utility.Vector2iVector(edges))
  line_set.colors = o3d.utility.Vector3dVector(colors)

  return line_set, icp_line_set, pcd_combined_down

def compute_distance(GT, ICP):
  GT_array = np.asarray(GT.points)
  ICP_array = np.asarray(ICP.points)
  
  with open('./distance.txt', 'w') as f:
    for i in range(len(GT_array)):
      gt_no_scale = ((GT_array[i]*1000.)/255.)*10.
      icp_no_scale = ((ICP_array[i]*1000.)/255.)*10.
      dis = distance.euclidean(gt_no_scale, icp_no_scale)

      f.write(f'Step{i+1: 4d}: Groundtruth: {gt_no_scale}, Estimated: {icp_no_scale}, Distance: {dis: .8f}\n')

line_set, icp_line_set, pcd_combined_down = test_GT(icp_matrix, pcd_combined_down)
# compute_distance(line_set, icp_line_set)
print('\nfinish adding trajectory')


"""show no roof result"""
def remove_roof(source, dataset = dataset_name):
    points = np.asarray(source.points)
    colors = np.asarray(source.colors)
    x = points[:, 0]
    y = points[:, 1]
    z = points[:, 2]
    r = colors[:, 0]
    g = colors[:, 1]
    b = colors[:, 2]

    upper_bound = np.max(y) - 0.06
    valid = y < upper_bound

    X = x[valid]
    Y = y[valid]
    Z = z[valid]
    R = r[valid]
    G = g[valid]
    B = b[valid]
 
    position = np.transpose(np.vstack((X, Y, Z, )))
    color = np.transpose(np.vstack((R, G, B)))

    new = o3d.geometry.PointCloud()
    new.points = o3d.utility.Vector3dVector(position[:, 0:3])
    new.colors = o3d.utility.Vector3dVector(color[:, 0:3])

    return new

def visualization_without_roof(line_set, icp_line_set, pcd_combined_down):
  no_roof = remove_roof(pcd_combined_down)
  
  print('\nshowing no roof result')
  o3d.visualization.draw_geometries([line_set, icp_line_set, no_roof])

visualization_without_roof(line_set, icp_line_set, pcd_combined_down)

"""show 2D result"""
def transform_3D_to_2D(source):
  points = np.asarray(source.points)
  colors = np.asarray(source.colors)
  x = points[:, 0]
  y = points[:, 1]
  z = points[:, 2]
  r = colors[:, 0]
  g = colors[:, 1]
  b = colors[:, 2]

  lower_bound = min(y) + 0.032
  upper_bound = max(y) - 0.04

  valid = y < upper_bound

  X = x[valid]
  Y = y[valid]
  Z = z[valid]
  R = r[valid]
  G = g[valid]
  B = b[valid]

  for i in range(len(Y)):
    Y[i] = 0.05

  position = np.transpose(np.vstack((X, Y, Z, )))
  color = np.transpose(np.vstack((R, G, B)))

  new = o3d.geometry.PointCloud()
  new.points = o3d.utility.Vector3dVector(position[:, 0:3])
  new.colors = o3d.utility.Vector3dVector(color[:, 0:3])

  return new

def visualizatio_2D(line_set, icp_line_set, pcd_combined_down):
  twoD_result = transform_3D_to_2D(pcd_combined_down)

  print('\nshowing 2D result')
  o3d.visualization.draw_geometries([line_set, icp_line_set, twoD_result])

# visualizatio_2D(line_set, icp_line_set, pcd_combined_down)
print('\nfinish executing\n')
