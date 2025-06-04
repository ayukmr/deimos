import sys

import numpy as np
from PIL import Image
from matplotlib import pyplot as plt

from skimage.measure import find_contours
from skimage.morphology import binary_closing, disk

THRESHOLD = 300
CLOSED = 3
AVERAGE = 5

MAX_PTS = 1000

SX = np.array([
    [1, 0, -1],
    [2, 0, -2],
    [1, 0, -1]
])
SY = np.array([
    [ 1,  2,  1],
    [ 0,  0,  0],
    [-1, -2, -1],
])

def main():
    if len(sys.argv) != 2:
        print('error: incorrect arguments')
        exit(1)

    binary = detect_edges(sys.argv[1])
    regions = analyze_regions(binary)

    kernel = np.ones(AVERAGE) / AVERAGE

    for center, points in regions:
        points = angle_sort(points, center)

        cy, cx = center
        py, px = points.T

        px = np.convolve(px, kernel, mode='valid')
        py = np.convolve(py, kernel, mode='valid')

        x_eq = fft_equation(px)
        y_eq = fft_equation(py)

        if x_eq and y_eq:
            print(f'({cx:.3f}+{x_eq}, -({cy:.3f}+{y_eq}))')

def detect_edges(path):
    pil = Image.open(path).convert('L')
    img = np.array(pil).astype(float)

    padded = np.pad(img, 1)
    grad = np.zeros_like(img)

    rows, cols = img.shape

    for y in range(rows):
        for x in range(cols):
            region = padded[y:y + 3, x:x + 3]

            gx = np.sum(SX * region)
            gy = np.sum(SY * region)

            grad[y, x] = np.hypot(gx, gy)

    binary = grad > THRESHOLD

    binary = binary_closing(
        binary,
        footprint=disk(CLOSED)
    )

    return binary

def analyze_regions(binary):
    areas = ~binary

    contours = find_contours(
        areas.astype(float),
        level=0.5
    )

    centers = []
    edges = []

    for contour in contours:
        if len(contour) == 0:
            continue

        pts = np.round(contour).astype(int)
        pts = interpolate(pts)

        while len(pts) > MAX_PTS:
            pts = pts[::2]

        center = np.mean(pts, axis=0)

        centers.append(center)
        edges.append(pts)

    show_points(binary, centers, edges)

    return list(zip(centers, edges))

def fft_equation(data):
    harmonics = np.fft.fft(data)

    n = len(data)
    terms = []

    for k in range(1, n // 2):
        a =  2 * np.real(harmonics[k]) / n
        b = -2 * np.imag(harmonics[k]) / n

        if abs(a) > 1e-3:
            terms.append(f'{a:.3f}*\\cos({k}t*2\\pi)')

        if abs(b) > 1e-3:
            terms.append(f'{b:.3f}*\\sin({k}t*2\\pi)')

    return '+'.join(terms)

def show_points(binary, centers, edges):
    img = Image.fromarray(binary.astype(np.uint8) * 255)

    plt.figure(figsize=(6, 6))
    plt.imshow(img, cmap='gray')
    plt.axis('off')

    for cy, cx in centers:
        plt.scatter(cx, cy, color='red', s=30)

    for pts in edges:
        if pts.size > 0:
            ey, ex = pts.T
            plt.scatter(ex, ey, color='blue', s=1)

    plt.show()

def interpolate(pts):
    new = []

    for i in range(len(pts)):
        a = pts[i]
        b = pts[(i + 1) % len(pts)]

        avg = (a + b) / 2

        new.append(a)
        new.append(avg)

    return np.round(new).astype(int)

def angle_sort(points, center):
    cy, cx = center

    dx = points[:, 1] - cx
    dy = points[:, 0] - cy

    angles = np.arctan2(dy, dx)
    angles = (angles + 2 * np.pi) % (2 * np.pi)

    return points[np.argsort(angles)]
