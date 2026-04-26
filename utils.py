import math


def haversine_distance(coord1, coord2):
    """
    计算两经纬度坐标点之间的距离 (单位: km)
    coord1, coord2 格式为 (纬度 latitude, 经度 longitude)
    """
    lat1, lon1 = coord1
    lat2, lon2 = coord2
    R = 6371.0 # 地球平均半径 (公里)

    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    delta_phi = math.radians(lat2 - lat1)
    delta_lambda = math.radians(lon2 - lon1)

    a = math.sin(delta_phi / 2.0)**2 + \
        math.cos(phi1) * math.cos(phi2) * math.sin(delta_lambda / 2.0)**2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))

    distance = R * c
    return distance


def _hungarian_minimize(cost):
    n = len(cost)
    m = len(cost[0]) if n else 0
    if n == 0 or m == 0:
        return []

    u = [0.0] * (n + 1)
    v = [0.0] * (m + 1)
    p = [0] * (m + 1)
    way = [0] * (m + 1)

    for i in range(1, n + 1):
        p[0] = i
        j0 = 0
        minv = [float("inf")] * (m + 1)
        used = [False] * (m + 1)

        while True:
            used[j0] = True
            i0 = p[j0]
            delta = float("inf")
            j1 = 0
            for j in range(1, m + 1):
                if not used[j]:
                    cur = cost[i0 - 1][j - 1] - u[i0] - v[j]
                    if cur < minv[j]:
                        minv[j] = cur
                        way[j] = j0
                    if minv[j] < delta:
                        delta = minv[j]
                        j1 = j
            for j in range(m + 1):
                if used[j]:
                    u[p[j]] += delta
                    v[j] -= delta
                else:
                    minv[j] -= delta
            j0 = j1
            if p[j0] == 0:
                break

        while True:
            j1 = way[j0]
            p[j0] = p[j1]
            j0 = j1
            if j0 == 0:
                break

    pairs = []
    for j in range(1, m + 1):
        if p[j] != 0:
            pairs.append((p[j] - 1, j - 1))
    return pairs


def hungarian_assignment(cost_matrix):
    """
    Minimum-cost assignment for rectangular matrix.
    Returns list of (row_idx, col_idx).
    """
    rows = len(cost_matrix)
    cols = len(cost_matrix[0]) if rows else 0
    if rows == 0 or cols == 0:
        return []

    if rows <= cols:
        return _hungarian_minimize(cost_matrix)

    transposed = [[cost_matrix[r][c] for r in range(rows)] for c in range(cols)]
    pairs_t = _hungarian_minimize(transposed)
    return [(c, r) for r, c in pairs_t]
