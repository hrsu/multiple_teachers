from numpy import *
import scipy.spatial.distance as dist  # 导入scipy距离公式

#欧式距离
def Euclidean_Distance(a, b):
    vector1 = mat(a)
    vector2 = mat(b)
    dis_Euclidean = float(sqrt((vector1 - vector2) * ((vector1 - vector2).T)))
    print("Euclidean_Distance (欧式距离): ", dis_Euclidean)
    return dis_Euclidean

#曼哈顿距离
def Manhattan_Distance(a, b):
    vector1 = mat(a)
    vector2 = mat(b)
    dis_Manhattan = sum(abs(vector1-vector2))
    print("Manhattan_Distance (曼哈顿距离): ", dis_Manhattan)
    return dis_Manhattan

#切比雪夫距离
def Chebyshev_Distance(a, b):
    vector1 = mat(a)
    vector2 = mat(b)
    dis_Chebyshev = abs(vector1-vector2).max()
    print("Chebyshev_Distance (切比雪夫距离): ", dis_Chebyshev)
    return dis_Chebyshev

#夹角余弦
def Cosine(a, b1):
    vector1 = mat(a)
    b = np.array(b1).reshape(len(b1), 1)
    vector2 = mat(b)
    cosV12 = dot(vector1, vector2) / (linalg.norm(vector1) * linalg.norm(vector2))
    print("Cosine (夹角余弦): ", cosV12)
    return cosV12

#汉明距离
def Hamming_distance(a, b):
    matV = mat([a, b])
    smstr = nonzero(matV[0] - matV[1])
    dis_Hamming = shape(smstr[0])[1]
    print("Hamming_distance (汉明距离): ", dis_Hamming)
    return dis_Hamming

#杰卡德相似系数
def Jaccard_similarity_coefficient(a, b):
    matV = mat([a, b])
    dis_Jaccard = dist.pdist(matV, 'jaccard')
    print("dist.jaccard (杰卡德相似系数): ", dis_Jaccard)

if __name__ == '__main__':
    a = [1, 2, 3]
    b = [4, 5, 6]
    Euclidean_Distance(a, b)
    Manhattan_Distance(a, b)
    Chebyshev_Distance(a, b)
    Cosine(a, b)
    Hamming_distance(a, b)
    Jaccard_similarity_coefficient(a, b)

