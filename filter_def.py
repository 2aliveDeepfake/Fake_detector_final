import cv2
import numpy as np

# 눈 필터
def eye(image) :
    tmp = image.copy()

    # Emboss
    tmp = cv2.cvtColor(tmp, cv2.COLOR_RGB2GRAY)
    Kernel_emboss = np.array([[-2, -1, 0], [-1, 1, 1], [0, 1, 2]])
    tmp = cv2.filter2D(tmp, cv2.CV_8U, Kernel_emboss)
    tmp = cv2.cvtColor(tmp, cv2.COLOR_GRAY2RGB)

    # Histogram_Equalization
    tmp = cv2.cvtColor(tmp, cv2.COLOR_RGB2GRAY)
    cv2.equalizeHist(tmp, tmp)
    tmp = cv2.cvtColor(tmp, cv2.COLOR_GRAY2RGB)

    # # Bilateral
    tmp = cv2.bilateralFilter(tmp, 19, 153, 102)
    tmp = cv2.cvtColor(tmp, cv2.COLOR_RGB2GRAY)

    # Unsharp

    G_ksize = 19
    sigmaX = 0
    sigmaY = 0
    alpha = 10
    beta = 5
    gamma = 0

    # 필터는 홀수여야한다.

    if (G_ksize % 2 == 1):
        tmp2 = cv2.GaussianBlur(tmp, (G_ksize, G_ksize), sigmaX / 10, sigmaY / 10)
        work = cv2.addWeighted(tmp, alpha / 10, tmp2, -1 + (beta / 10), gamma)

    # 필터 통과한 이미지 변수에 넣기
    # work = tmp.copy()
    work = cv2.cvtColor(work, cv2.COLOR_GRAY2RGB)

    # 필터 통과한 이미지 변수에 넣기
    return work


# 코 필터
def nose(image) :
    tmp = image.copy()

    # Gamma_correction
    gamma = 20
    lookUpTable = np.empty((1, 256), np.uint8)
    for j in range(256):
        lookUpTable[0, j] = np.clip(pow(j / 255.0, 1.0 / (gamma / 10)) * 255.0, 0, 255)

    tmp = cv2.LUT(tmp, lookUpTable)

    #CLAHE
    tmp = cv2.cvtColor(tmp, cv2.COLOR_RGB2Lab)
    l, a, b = cv2.split(tmp)
    clahe = cv2.createCLAHE(clipLimit=0, tileGridSize=(10, 10))
    dst = clahe.apply(l)
    l = dst.copy()
    tmp = cv2.merge((l, a, b))
    tmp = cv2.cvtColor(tmp, cv2.COLOR_LAB2RGB)

    #THRESH_TRUNC
    tmp = cv2.cvtColor(tmp, cv2.COLOR_RGB2GRAY)
    ret, tmp = cv2.threshold(tmp, 180, 255, cv2.THRESH_TRUNC)
    tmp = cv2.cvtColor(tmp, cv2.COLOR_GRAY2RGB)
    image = tmp.copy()
    return image

# 입 필터
def mouth(image) :
    tmp = image.copy()

    # Gamma_correction
    gamma = 7
    lookUpTable = np.empty((1, 256), np.uint8)
    for j in range(256):
        lookUpTable[0, j] = np.clip(pow(j / 255.0, 1.0 / (gamma / 10)) * 255.0, 0, 255)

    tmp = cv2.LUT(tmp, lookUpTable)

    # CLAHE
    tmp = cv2.cvtColor(tmp, cv2.COLOR_RGB2Lab)
    l, a, b = cv2.split(tmp)
    clahe = cv2.createCLAHE(clipLimit=1, tileGridSize=(40, 40))
    dst = clahe.apply(l)
    l = dst.copy()
    tmp = cv2.merge((l, a, b))
    tmp = cv2.cvtColor(tmp, cv2.COLOR_LAB2RGB)

    # THRESH_TRUNC
    tmp = cv2.cvtColor(tmp, cv2.COLOR_RGB2GRAY)
    ret, tmp = cv2.threshold(tmp, 255, 255, cv2.THRESH_TRUNC)
    # 3채널이미지로 바꿔준다.
    tmp = cv2.cvtColor(tmp, cv2.COLOR_GRAY2RGB)

    # 필터 통과한 이미지 변수에 넣기
    image = tmp.copy()

    return image

# 얼굴 가로 세로 노이즈
def face_gridnoise(image) :
    tmp = image.copy()
    gray = cv2.cvtColor(tmp, cv2.COLOR_BGR2GRAY)

    # # #emboss필터 적용
    tmp = gray.copy()
    # work = tmp.copy()
    Kernel = np.array([[-11, -1, 0], [-1, 1, 1], [0, 1, 11]])
    work = cv2.filter2D(tmp, cv2.CV_8U, Kernel)
    work = cv2.cvtColor(work, cv2.COLOR_GRAY2RGB)

    # 필터 통과한 이미지 변수에 넣기
    image = work.copy()
    return image

# 얼굴 점 노이즈
def face_dotnoise(image) :
    tmp = image.copy()
    tmp = cv2.cvtColor(tmp, cv2.COLOR_RGB2GRAY)

    # Emboss
    Kernel = np.array([[-11, -1, 0], [-1, 1, 1], [0, 1, 11]])
    tmp = cv2.filter2D(tmp, cv2.CV_8U, Kernel)

    # Median
    tmp = cv2.medianBlur(tmp, ksize=5)

    # Laplacian_of_Gaussian
    tmp = cv2.GaussianBlur(tmp, ksize=(5, 5), sigmaX=0, sigmaY=0, borderType=cv2.BORDER_DEFAULT)
    tmp = cv2.Laplacian(tmp, cv2.CV_16S, ksize=1, scale=4, delta=0,
                        borderType=cv2.BORDER_DEFAULT)
    LaplacianImage = cv2.convertScaleAbs(tmp)
    tmp = cv2.cvtColor(LaplacianImage, cv2.COLOR_GRAY2RGB)

    # THRESH_BINARY
    ret, tmp = cv2.threshold(tmp, 167, 255, cv2.THRESH_BINARY)
    image = tmp.copy()
    return image

# 안경 필터
def eye_glasses(image) :
    tmp = image.copy()

    # Gamma_correction (43)
    gamma = 20
    lookUpTable = np.empty((1, 256), np.uint8)
    for j in range(256):
        lookUpTable[0, j] = np.clip(pow(j / 255.0, 1.0 / (gamma / 10)) * 255.0, 0, 255)

    tmp = cv2.LUT(tmp, lookUpTable)

    # High_pass (28)
    ksize_26 = 6
    kernel = np.array([[-0.5, -1, -0.5], [-1, ksize_26, -1], [-0.5, -1, -0.5]])
    tmp = cv2.filter2D(tmp, cv2.CV_8U, kernel)

    # Glasses From Paper
    # Laple_Kernel = np.array([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]])
    Gaussian_Kernel = np.array([[1, 2, 1], [2, 4, 2], [1, 2, 1]])
    # tmp = cv2.filter2D(tmp, cv2.CV_16S, Laple_Kernel)
    tmp = cv2.filter2D(tmp, cv2.CV_16S, Gaussian_Kernel)
    # tmp = cv2.filter2D(tmp, cv2.CV_16S, Laple_Kernel)
    tmp = np.clip(tmp, 0, 255)
    tmp = np.uint8(tmp)

    work = cv2.cvtColor(tmp, cv2.COLOR_RGB2GRAY)
    image = cv2.cvtColor(work, cv2.COLOR_GRAY2RGB)

    return image

# 미간에 콧구멍
def nose_in_b(image) :
    tmp = image.copy()
    # THRESH_BINARY
    tmp = cv2.cvtColor(tmp, cv2.COLOR_RGB2GRAY)
    ret, tmp = cv2.threshold(tmp, 55, 255, cv2.THRESH_BINARY)

    ###### could not broadcast input array from shape 에러시
    # 차원 관련 문제니 gray2rgb 적용
    tmp = cv2.cvtColor(tmp, cv2.COLOR_GRAY2RGB)

    # 필터 통과한 이미지 변수에 넣기
    image = tmp.copy()
    return image


# 코 노이즈
def nose_noise(image) :
    tmp = image.copy()

    gray = cv2.cvtColor(tmp, cv2.COLOR_BGR2GRAY)

    # # #emboss필터 적용
    tmp = gray.copy()
    # work = tmp.copy()
    Kernel = np.array([[-11, -1, 0], [-1, 1, 1], [0, 1, 11]])
    work = cv2.filter2D(tmp, cv2.CV_8U, Kernel)
    work = cv2.cvtColor(work, cv2.COLOR_GRAY2RGB)

    # 필터 통과한 이미지 변수에 넣기
    image = work.copy()

    return image
