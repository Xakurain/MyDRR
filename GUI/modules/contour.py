import torch
import torch.nn.functional as F
import numpy as np
import cv2
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def gaussian_kernel(size, sigma):
    kernel = torch.tensor(
        [[np.exp(-(x ** 2 + y ** 2) / (2 * sigma ** 2)) for x in range(-size // 2 + 1, size // 2 + 1)] for y in
         range(-size // 2 + 1, size // 2 + 1)])
    kernel = kernel / torch.sum(kernel)
    kernel = kernel.float()
    return kernel
def gaussian_blur(image, kernel_size, sigma):
    kernel = gaussian_kernel(kernel_size, sigma).to(device)
    kernel = kernel.view(1, 1, kernel_size, kernel_size)
    image = image.unsqueeze(0).unsqueeze(0)
    blurred_image = F.conv2d(image, kernel, padding=kernel_size // 2)
    return blurred_image.squeeze()

def sobel_edge_detection(image):
    with torch.no_grad():
        Kx = torch.tensor([[1, 0, -1], [2, 0, -2], [1, 0, -1]], dtype=torch.float32, device=device)
        Ky = torch.tensor([[1, 2, 1], [0, 0, 0], [-1, -2, -1]], dtype=torch.float32, device=device)

        Gx = F.conv2d(image.unsqueeze(0).unsqueeze(0), Kx.view(1, 1, 3, 3))
        Gy = F.conv2d(image.unsqueeze(0).unsqueeze(0), Ky.view(1, 1, 3, 3))

        G = torch.sqrt(Gx ** 2 + Gy ** 2).squeeze()
        G = (G / G.max()) * 255
        return G.byte()


def prewitt_edge_detection(image):
    with torch.no_grad():
        Kx = torch.tensor([[1, 0, -1], [1, 0, -1], [1, 0, -1]], dtype=torch.float32, device=device)
        Ky = torch.tensor([[1, 1, 1], [0, 0, 0], [-1, -1, -1]], dtype=torch.float32, device=device)

        Gx = F.conv2d(image.unsqueeze(0).unsqueeze(0), Kx.view(1, 1, 3, 3))
        Gy = F.conv2d(image.unsqueeze(0).unsqueeze(0), Ky.view(1, 1, 3, 3))

        G = torch.sqrt(Gx ** 2 + Gy ** 2).squeeze()
        G = (G / G.max()) * 255
        return G.byte()

def canny_edge_detection(image, low_threshold=10, high_threshold=30):
    with torch.no_grad():
        # smoothed_image = gaussian_blur(image, kernel_size=5, sigma=1.4)

        smoothed_image = image
        Kx = torch.tensor([[1, 0, -1], [2, 0, -2], [1, 0, -1]], dtype=torch.float32, device=device)
        Ky = torch.tensor([[1, 2, 1], [0, 0, 0], [-1, -2, -1]], dtype=torch.float32, device=device)
        Gx = F.conv2d(smoothed_image.unsqueeze(0).unsqueeze(0), Kx.view(1, 1, 3, 3))
        Gy = F.conv2d(smoothed_image.unsqueeze(0).unsqueeze(0), Ky.view(1, 1, 3, 3))
        G = torch.sqrt(Gx ** 2 + Gy ** 2).squeeze()
        G = (G / G.max()) * 255

        theta = torch.atan2(Gy, Gx).squeeze() * 180 / np.pi
        theta[theta < 0] += 180

        nms = torch.zeros_like(G)
        rows, cols = G.shape
        for i in range(1, rows - 1):
            for j in range(1, cols - 1):
                angle = theta[i, j]
                if (0 <= angle < 22.5) or (157.5 <= angle <= 180):
                    neighbors = [G[i, j + 1], G[i, j - 1]]
                elif 22.5 <= angle < 67.5:
                    neighbors = [G[i + 1, j - 1], G[i - 1, j + 1]]
                elif 67.5 <= angle < 112.5:
                    neighbors = [G[i + 1, j], G[i - 1, j]]
                else:
                    neighbors = [G[i + 1, j + 1], G[i - 1, j - 1]]

                if G[i, j] >= max(neighbors):
                    nms[i, j] = G[i, j]

        high_threshold_value = nms.max() * high_threshold / 255
        low_threshold_value = high_threshold_value * low_threshold / 100

        strong_edges = (nms >= high_threshold_value).int()
        weak_edges = ((nms >= low_threshold_value) & (nms < high_threshold_value)).int()

        edges = torch.zeros_like(nms)
        edges[strong_edges == 1] = 255

        def track_edges(i, j):
            if weak_edges[i, j] == 1:
                edges[i, j] = 255
                weak_edges[i, j] = 0
                if i > 0: track_edges(i - 1, j)
                if i < rows - 1: track_edges(i + 1, j)
                if j > 0: track_edges(i, j - 1)
                if j < cols - 1: track_edges(i, j + 1)
                if i > 0 and j > 0: track_edges(i - 1, j - 1)
                if i > 0 and j < cols - 1: track_edges(i - 1, j + 1)
                if i < rows - 1 and j > 0: track_edges(i + 1, j - 1)
                if i < rows - 1 and j < cols - 1: track_edges(i + 1, j + 1)

        for i in range(rows):
            for j in range(cols):
                if strong_edges[i, j] == 1:
                    track_edges(i, j)

        return edges.byte()
def roberts_cross_operator(image):
    with torch.no_grad():
        Kx = torch.tensor([[1, 0],
                           [0, -1]], dtype=torch.float32, device=device)

        Ky = torch.tensor([[0, 1],
                           [-1, 0]], dtype=torch.float32, device=device)

        Gx = F.conv2d(image.unsqueeze(0).unsqueeze(0), Kx.view(1, 1, 2, 2), padding=0)
        Gy = F.conv2d(image.unsqueeze(0).unsqueeze(0), Ky.view(1, 1, 2, 2), padding=0)

        edges = torch.sqrt(Gx ** 2 + Gy ** 2).squeeze()
        edges = (edges / edges.max()) * 255
        return edges.byte()


def kirsch_operator(image):
    with torch.no_grad():
        kernels = [
            torch.tensor([[5, 5, 5],
                          [-3, 0, -3],
                          [-3, -3, -3]], dtype=torch.float32, device=device),

            torch.tensor([[-3, 5, 5],
                          [-3, 0, 5],
                          [-3, -3, -3]], dtype=torch.float32, device=device),

            torch.tensor([[-3, -3, 5],
                          [-3, 0, 5],
                          [-3, -3, 5]], dtype=torch.float32, device=device),

            torch.tensor([[-3, -3, -3],
                          [-3, 0, 5],
                          [-3, 5, 5]], dtype=torch.float32, device=device),

            torch.tensor([[-3, -3, -3],
                          [-3, 0, -3],
                          [5, 5, 5]], dtype=torch.float32, device=device),

            torch.tensor([[-3, -3, -3],
                          [5, 0, -3],
                          [5, 5, -3]], dtype=torch.float32, device=device),

            torch.tensor([[5, -3, -3],
                          [5, 0, -3],
                          [5, -3, -3]], dtype=torch.float32, device=device),

            torch.tensor([[5, 5, -3],
                          [5, 0, -3],
                          [-3, -3, -3]], dtype=torch.float32, device=device)
        ]

        results = []
        for kernel in kernels:
            G = F.conv2d(image.unsqueeze(0).unsqueeze(0), kernel.view(1, 1, 3, 3), padding=1)
            G = torch.abs(G).squeeze()
            results.append(G)

        edges = torch.max(torch.stack(results), dim=0)[0]
        edges = (edges / edges.max()) * 255
        return edges.byte()

def otsu_thresholding(image):
    histogram, _ = np.histogram(image.ravel(), bins=256, range=(0, 256))
    total = image.size
    current_max, threshold = 0, 0
    sumT, sumF, weightB, weightF = 0, 0, 0, 0

    for i in range(256):
        sumT += i * histogram[i]

    for i in range(256):
        weightB += histogram[i]
        if weightB == 0:
            continue
        weightF = total - weightB
        if weightF == 0:
            break
        sumF += i * histogram[i]
        sumB = sumT - sumF
        meanB = sumB / weightB
        meanF = sumF / weightF
        between = weightB * weightF * (meanB - meanF) ** 2

        if between > current_max:
            current_max = between
            threshold = i

    _, binary_image = cv2.threshold(image, threshold, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    return binary_image

def distance_transform(binary_image):
    rows, cols = binary_image.shape
    dist = np.zeros_like(binary_image, dtype=np.float32)

    for i in range(rows):
        for j in range(cols):
            if binary_image[i, j] == 0:
                dist[i, j] = 0
            else:
                dist[i, j] = np.min([dist[i - 1, j] + 1 if i - 1 >= 0 else np.inf,
                                     dist[i, j - 1] + 1 if j - 1 >= 0 else np.inf,
                                     dist[i - 1, j - 1] + 1 if i - 1 >= 0 and j - 1 >= 0 else np.inf])

    for i in range(rows - 1, -1, -1):
        for j in range(cols - 1, -1, -1):
            if binary_image[i, j] == 1:
                dist[i, j] = np.min([dist[i, j],
                                     dist[i + 1, j] + 1 if i + 1 < rows else np.inf,
                                     dist[i, j + 1] + 1 if j + 1 < cols else np.inf,
                                     dist[i + 1, j + 1] + 1 if i + 1 < rows and j + 1 < cols else np.inf])

    return dist


def watershed(binary_image):
    dist = distance_transform(binary_image)
    markers = np.zeros_like(binary_image, dtype=np.int32)
    markers[dist > 0.7 * dist.max()] = 1
    markers[dist <= 0.7 * dist.max()] = 2

    rows, cols = binary_image.shape
    for i in range(rows):
        for j in range(cols):
            if markers[i, j] == 1:
                binary_image[i, j] = 255
            else:
                binary_image[i, j] = 0

    return binary_image

def map_binary_to_original(original_image, binary_image):
    # 生成掩码
    mask = binary_image.copy()
    mask[mask > 0] = 255  # 将非零值设为255

    # 将掩码应用于原始灰度图像
    result_image = cv2.bitwise_and(original_image, mask)

    return result_image

# 示例用法
if __name__ == "__main__":
    image = cv2.imread('y_pred_16458.png')
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # # 应用Otsu阈值分割
    binary_image = otsu_thresholding(image)
    # _, binary_image = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    # 映射到原始图像
    mapped_image = map_binary_to_original(image, binary_image)

    # 显示结果
    cv2.imshow('Original Image', image)
    cv2.imshow('Binary Image (Otsu)', binary_image)
    cv2.imshow('mapped image', mapped_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    # image = watershed(image)

    segmented_image = torch.from_numpy(mapped_image).float().to(device)
    # Sobel 边缘检测
    sobel_edges = sobel_edge_detection(segmented_image)
    cv2.imshow('2', sobel_edges.cpu().numpy())
    cv2.waitKey(0)

    # Prewitt 边缘检测
    prewitt_edges = prewitt_edge_detection(segmented_image)
    cv2.imshow('3', prewitt_edges.cpu().numpy())
    cv2.waitKey(0)

    # Roberts Cross 算子边缘检测
    roberts_edges = roberts_cross_operator(segmented_image)
    cv2.imshow("Roberts Cross Edges", roberts_edges.cpu().numpy())
    cv2.waitKey(0)

    # Kirsch 算子边缘检测
    kirsch_edges = kirsch_operator(segmented_image)
    cv2.imshow("Kirsch Edges", kirsch_edges.cpu().numpy())
    cv2.waitKey(0)

    # Canny 边缘检测
    canny_edges = canny_edge_detection(segmented_image)
    cv2.imshow('4', canny_edges.cpu().numpy())
    cv2.waitKey(0)

    # 清理显存
    del segmented_image, sobel_edges, prewitt_edges, canny_edges
    torch.cuda.empty_cache()