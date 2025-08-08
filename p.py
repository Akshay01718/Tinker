import numpy as np
import cv2
import torch
import urllib.request
from queue import PriorityQueue
import matplotlib.pyplot as plt

# Load MiDaS model and transformations
midas = torch.hub.load('intel-isl/MiDaS', 'MiDaS_small')
midas.eval()
midas_transforms = torch.hub.load('intel-isl/MiDaS', 'transforms')
transform = midas_transforms.small_transform

# Load image from local path or URL
def load_image(image_path_or_url):
    if image_path_or_url.startswith('http'):
        resp = urllib.request.urlopen(image_path_or_url)
        image = np.asarray(bytearray(resp.read()), dtype="uint8")
        image = cv2.imdecode(image, cv2.IMREAD_COLOR)
    else:
        image = cv2.imread(image_path_or_url)
    if image is None:
        raise ValueError(f"Failed to load image from {image_path_or_url}")
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Predict depth
def predict_depth(image):
    input_batch = transform(image).unsqueeze(0)
    if input_batch.dim() == 5:
        input_batch = input_batch.squeeze(1)
    with torch.no_grad():
        prediction = midas(input_batch)
        prediction = torch.nn.functional.interpolate(
            prediction.unsqueeze(1),
            size=image.shape[:2],
            mode='bicubic',
            align_corners=False
        ).squeeze()
    depth_map = prediction.cpu().numpy()
    return depth_map

# Normalize depth map to [0, 1]
def normalize_depth(depth_map):
    depth_min = depth_map.min()
    depth_max = depth_map.max()
    return (depth_map - depth_min) / (depth_max - depth_min)

# Create cost map with high cost for puddles (deeper areas)
def create_cost_map(depth_map, threshold=0.3):
    norm_depth = normalize_depth(depth_map)
    cost_map = np.ones_like(norm_depth)
    cost_map[norm_depth < threshold] = 0.01  # low cost for dry areas
    cost_map[norm_depth >= threshold] = 10    # high cost for puddles
    return cost_map

# Implement A* pathfinding
def astar(cost_map, start, end):
    neighbors = [(0,1),(0,-1),(1,0),(-1,0),(1,1),(1,-1),(-1,1),(-1,-1)]
    h, w = cost_map.shape
    close_set = set()
    came_from = {}
    gscore = {start:0}
    fscore = {start: np.linalg.norm(np.array(start)-np.array(end))}
    oheap = PriorityQueue()
    oheap.put((fscore[start], start))

    while not oheap.empty():
        current = oheap.get()[1]
        if current == end:
            data = []
            while current in came_from:
                data.append(current)
                current = came_from[current]
            data.append(start)
            return data[::-1]

        close_set.add(current)
        for i, j in neighbors:
            neighbor = current[0]+i, current[1]+j
            if 0 <= neighbor[0] < h and 0 <= neighbor[1] < w:
                tentative_g_score = gscore[current] + cost_map[neighbor[0], neighbor[1]]
                if neighbor in close_set and tentative_g_score >= gscore.get(neighbor, 0):
                    continue

                if tentative_g_score < gscore.get(neighbor, float('inf')) or neighbor not in [i[1] for i in oheap.queue]:
                    came_from[neighbor] = current
                    gscore[neighbor] = tentative_g_score
                    fscore[neighbor] = tentative_g_score + np.linalg.norm(np.array(neighbor)-np.array(end))
                    oheap.put((fscore[neighbor], neighbor))
    return []

# Draw path on original image
def draw_path(image, path):
    image_out = image.copy()
    for i in range(1, len(path)):
        cv2.line(image_out, (path[i-1][1], path[i-1][0]), (path[i][1], path[i][0]), (255, 0, 0), 2)
    return image_out

# Main processing function
def process_puddle_path(image_path):
    image = load_image(image_path)
    depth_map = predict_depth(image)
    cost_map = create_cost_map(depth_map, threshold=0.3)

    h, w = cost_map.shape
    start = (h-1, w//2)  # bottom center of image
    end = (0, w//2)      # top center of image

    path = astar(cost_map, start, end)

    image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    image_with_path = draw_path(image_bgr, path)

    output_file = 'puddle_path_output.png'
    cv2.imwrite(output_file, image_with_path)
    cv2.imshow('Optimal Path Across Puddle', image_with_path)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    print(f"Saved output to {output_file}")

# Usage example with your attached input image path
input_image_path = 'https://media.istockphoto.com/id/689759240/photo/puddle-on-road-during-rain.jpg?s=612x612&w=0&k=20&c=t9AknYDIH-N2fp2zJTjmXiexXL2lJ-5noFPereDH8eE='
process_puddle_path(input_image_path)
