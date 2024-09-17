def solution(x, y):
    size = len(x)

    sr_x = sum(x) / size
    sr_y = sum(y) / size
    sr_y_kv = sum(y ** 2) / size
    sr_kv_y = sr_y ** 2
    sr_xy = sum(x * y) / size
    sr_x_kv = sum(x ** 2) / size
    sr_kv_x = sr_x ** 2

    k = (sr_xy - sr_x * sr_y) / (sr_x_kv - sr_kv_x)
    b = sr_y - k*sr_x

    sigma_k = (1 / (size ** 0.5)) * (((sr_y_kv - sr_kv_y) / (sr_x_kv - sr_kv_x) - k ** 2) ** 0.5)
    sigma_b = sigma_k * (sr_x_kv - sr_kv_x) ** 0.5

    return [k, b, sigma_k, sigma_b]

def slych_pogr(x):
    size = len(x)

    sr_x = sum(x) / size
    sigma = ((1 / size / (size - 1))*sum([(i - sr_x)**2 for i in x]))**0.5

    return sigma

