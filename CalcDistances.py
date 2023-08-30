import csv
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib

matplotlib.use("TkAgg")
from scipy import stats


def read_file(file_path):
    file_extension = os.path.splitext(file_path)[1].lower()
    if file_extension == '.xyz':
        return read_xyz(file_path)
    elif file_extension == '.csv':
        return read_csv(file_path)
    else:
        raise ValueError("Unsupported file type. Only .xyz and .csv files are supported.")


def read_xyz(file_path):
    with open(file_path, 'r') as file:
        data_points = []
        for line in file:
            values = line.strip().split()
            x, y, z = map(float, values[:3])
            data_points.append([x, y, z])
    return np.array(data_points)


def read_csv(file_path):
    data_points = np.genfromtxt(file_path, delimiter=',', usecols=(0, 1, 2))
    return data_points


def clean_data(data_points, zscore_threshold=3.0):
    # calculate Z-scores for each dimension
    # the z-score range chosen is from -3.0 to 3.0 because 99.7% of normally distributed data falls within this range
    zscores = np.abs(stats.zscore(data_points, axis=0))
    # identify outliers by comparing Z-scores to the threshold and filter them out
    is_outlier = np.any(zscores > zscore_threshold, axis=1)
    filtered_points = data_points[~is_outlier]
    if filtered_points.shape[0] == 0:
        raise ValueError("All data points are identified as outliers.")
    return filtered_points


def select_best_dimensions(datapoints):
    variances = np.var(datapoints, axis=0)
    # Sort in descending order
    sorted_indices = np.argsort(variances)[::-1]
    # Select the dimensions with highest variance
    best_dimensions = sorted_indices[:2]
    return best_dimensions


def project_to_dimensions(datapoints, dimensions):
    return datapoints[:, dimensions]


def find_middle_point(filtered_points):
    min_point = np.min(filtered_points, axis=0)
    max_point = np.max(filtered_points, axis=0)
    middle_point = (min_point + max_point) / 2.0
    return middle_point


def calculate_average_distances(datapoints, drone_pos):
    # calculate the distances from the drone position to each datapoint
    distances = np.linalg.norm(datapoints - drone_pos, axis=1)
    # calculate the angles between the drone position to each datapoint
    angles = np.arctan2(datapoints[:, 1] - drone_pos[1], datapoints[:, 0] - drone_pos[0])
    # convert the angles to radians.
    angles = (angles + 2 * np.pi) % (2 * np.pi)
    num_angles = 360
    avg_distances = np.full(num_angles, np.nan)
    count_datapoints_per_angle = np.zeros(num_angles, dtype=int)
    # calculate the average distance for each angle, ignoring singular datapoints
    for i in range(num_angles):
        indices = np.where(np.logical_and(angles >= i * np.pi / 180, angles < (i + 1) * np.pi / 180))
        count = len(indices[0])
        if count > 1:
            avg_distances[i] = np.mean(distances[indices])
        count_datapoints_per_angle[i] = count

    # save the results to a CSV file
    with open('result.csv', 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["angle", "count", "average_distance"])
        for i in range(num_angles):
            writer.writerow([i, count_datapoints_per_angle[i], avg_distances[i]])
    return avg_distances


def count_nan_dist_neighbors(nan_indices, nan_counts, avg_distances, buffer_size=10):
    array_length = len(avg_distances)
    for index in nan_indices:
        left_neighbor = None
        right_neighbor = None
        for offset in range(1, buffer_size + 1):
            left_offset_index = (index - offset) % array_length
            right_offset_index = (index + offset) % array_length
            if left_neighbor is None:
                if not np.isnan(avg_distances[left_offset_index]):
                    left_neighbor = left_offset_index
            if right_neighbor is None:
                if not np.isnan(avg_distances[right_offset_index]):
                    right_neighbor = right_offset_index
            if left_neighbor is not None and right_neighbor is not None:
                break
        if left_neighbor is None:
            left_neighbor = left_offset_index - 1
        if right_neighbor is None:
            right_neighbor = right_offset_index + 1
        nan_counts[index] = right_neighbor - left_neighbor - 1
    return nan_counts

# assuming that the real exit point will be where we will have the maximum number of neighboring angles that have
# avg_distance=nan
def find_exit(datapoints, drone_pos, avg_distances):
    angles = np.linspace(0, 2 * np.pi, len(avg_distances), endpoint=False)
    nan_indices = np.where(np.isnan(avg_distances))[0]
    # if there are no nan values, return None,None
    if len(nan_indices) == 0:
        return None, None
    # count neighbors with average distance = nan
    nan_counts = np.zeros(len(angles))
    nan_counts = count_nan_dist_neighbors(nan_indices, nan_counts, avg_distances)
    max_count_indices = np.where(nan_counts == np.max(nan_counts))[0]
    max_count_indices_with_nan = [i for i in max_count_indices if np.isnan(avg_distances[i])]
    # choose the middle index among max_count_indices_with_nan
    middle_index = max_count_indices_with_nan[len(max_count_indices_with_nan) // 2]
    exit_angle_index = middle_index
    exit_angle = angles[exit_angle_index]
    # calculate the exit point coordinates
    exit_x = drone_pos[0] + np.cos(exit_angle) * np.mean(np.linalg.norm(datapoints, axis=1))
    exit_y = drone_pos[1] + np.sin(exit_angle) * np.mean(np.linalg.norm(datapoints, axis=1))
    exit_point = np.array([exit_x, exit_y])
    exit_angle = np.array([exit_angle])
    return exit_point, exit_angle


def plot_datapoints(datapoints, middle_point, title):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    x = datapoints[:, 0]
    y = datapoints[:, 1]
    z = datapoints[:, 2]
    ax.scatter(x, y, z, c='blue', label='datapoints')
    ax.scatter(middle_point[0], middle_point[1], middle_point[2], c='red', s=100, label='middle point')
    ax.set_title(title)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.legend()
    plt.show()


def plot_2D_datapoints(datapoints, middle_point, avg_distances, exit_point, dimenstions):
    dim1 = datapoints[:, 0]
    dim2 = datapoints[:, 1]
    if dimenstions[0] == 0:
        dim1_label = 'X'
    elif dimenstions[0] == 1:
        dim1_label = 'Y'
    else:
        dim1_label = 'Z'

    if dimenstions[1] == 0:
        dim2_label = 'X'
    elif dimenstions[1] == 1:
        dim2_label = 'Y'
    else:
        dim2_label = 'Z'
    plt.scatter(dim1, dim2, c='blue', label='Datapoints')
    plt.scatter(middle_point[0], middle_point[1], c='red', s=100, label='middle point')
    # convert the average distances to angles
    angles = np.linspace(0, 2 * np.pi, len(avg_distances), endpoint=False)
    # filter out nan distances
    avg_dist_angles = angles[~np.isnan(avg_distances)]
    avg_distances = avg_distances[~np.isnan(avg_distances)]
    # calculate the endpoint of the line representing average distances
    avg_dist_line_end_x = middle_point[0] + np.cos(avg_dist_angles) * avg_distances
    avg_dist_line_end_y = middle_point[1] + np.sin(avg_dist_angles) * avg_distances
    plt.plot(avg_dist_line_end_x, avg_dist_line_end_y, c='green', label='average distances', linewidth=2)
    # check if there is an exit point
    if exit_point is not None:
        exit_x = exit_point[0]
        exit_y = exit_point[1]
        plt.scatter(exit_x, exit_y, marker='X', c='orange', s=100, label='exit points')
    plt.title('2D datapoints')
    plt.xlabel(dim1_label)
    plt.ylabel(dim2_label)
    plt.legend(loc='upper left', bbox_to_anchor=(1.02, 1), prop={'size': 10})
    plt.gcf().set_size_inches(8, 6)
    plt.tight_layout()
    plt.show()


def plot_bitmap(avg_distances, exit_angle):
    angles = np.linspace(0, 2 * np.pi, len(avg_distances), endpoint=False)
    plt.polar(angles, avg_distances)
    plt.title('Exit angle')
    if exit_angle is not None:
        exit_y = exit_angle
        plt.scatter(exit_angle, exit_y, marker='X', c='orange', s=100, label='exit direction')
    plt.legend()
    plt.show()


if __name__ == '__main__':
    # read datapoints into numpy array
    datapoints = read_file('input/drive-download-20230827T133312Z-001/1+5+6.xyz')
    # clean the data (filter out singular points)
    filtered_points = clean_data(datapoints)
    # find the middle point of the datapoints (where the drone is probably located)
    drone_pos = find_middle_point(filtered_points)
    # plot the original datapoints in 3D with the middle point marked
    plot_datapoints(datapoints, drone_pos, 'original datapoints')
    # plot the filtered datapoints in 3D with the middle point marked
    plot_datapoints(filtered_points, drone_pos, 'filtered datapoints')
    # find the best dimensions for projection
    best_dimensions = select_best_dimensions(filtered_points)
    # project the datapoints to the selected dimensions
    projected_data = project_to_dimensions(filtered_points, best_dimensions)
    drone_pos = find_middle_point(projected_data)
    # calculate the average distances bitmap
    avg_distances = calculate_average_distances(projected_data, drone_pos)
    # find the exit point and exit angle.
    exit_point, exit_angle = find_exit(projected_data, drone_pos, avg_distances)
    # plot a 2D representation of the projected datapoints
    plot_2D_datapoints(projected_data, drone_pos, avg_distances, exit_point, best_dimensions)
    # plot the average distances with the exit point marked
    plot_bitmap(avg_distances, exit_angle)
    # print exit point and angle(in degrees)
    print(f"Exit Points: {exit_point}")
    exit_angle_in_degrees = np.degrees(exit_angle)
    print(f"Exit Angle: {exit_angle_in_degrees}")
