from PIL import Image, ImageDraw, ImageFont
import numpy as np
import matplotlib.pyplot as plt
from io import BytesIO

def plotSingularValues(val):
    plt.semilogy(val, 'k-')
    plt.show()

def SVD_Example1():
    example = np.random.randint(-10, 10, (2, 3))
    print(f"""Shape of example matrix, E: {example.shape}
Example matrix, E =
{np.array_str(example)}
""")
    
    p, d, q = np.linalg.svd(example)
    print("""E = PDQ,
    where: P and Q are unitary matrices (in this case, orthogonal matrices).
           D is a {example.shape} matrix with singular values of E as diagonal entries.
""")
    plotSingularValues(d)
    d = np.hstack((np.diag(d), np.zeros((2, 1))))
    print(f"""Shape of P: {p.shape}
P =
{np.array_str(p)}""")
    print(f"""Shape of D: {d.shape}
D =
{np.array_str(d)}""")
    print(f"""Shape of Q: {q.shape}
Q =
{np.array_str(q)}
""")
    computed = p @ d @ q
    print(f"PDQ -> {p.shape} x {d.shape} x {q.shape}")
    print(f"Shape of PDQ -> {computed.shape}")
    print(f"np.allclose(example, P @ D @ Q) = {np.allclose(example, computed)}")

def SVD_Example2():
    example = np.random.randint(-10, 10, (3, 2))
    print(f"""Shape of example matrix, E: {example.shape}
Example matrix, E =
{np.array_str(example)}
""")
    
    p, d, q = np.linalg.svd(example)
    d = np.vstack((np.diag(d), np.zeros((1,2))))
    print(f"""E = PDQ,
    where: P and Q are unitary matrices (in this case, orthogonal matrices).
           D is a {example.shape} matrix with singular values of E as diagonal entries.
""")
    print(f"""Shape of P: {p.shape}
P =
{np.array_str(p)}""")
    print(f"""Shape of D: {d.shape}
D =
{np.array_str(d)}""")
    print(f"""Shape of Q: {q.shape}
Q =
{np.array_str(q)}
""")
    computed = p @ d @ q
    print(f"PDQ -> {p.shape} x {d.shape} x {q.shape}")
    print(f"Shape of PDQ -> {computed.shape}")
    print(f"np.allclose(example, P @ D @ Q) = {np.allclose(example, computed)}")

# finding all orthonormal vectors
def findingOrthonormal():
    n, m = 6, 2  # for example
    mat = np.random.uniform(size=(n, m))
    x, y, z = np.linalg.svd(mat)
    print(f"""M = XYZ,
    where: X and Z are unitary matrices (in this case, orthogonal matrices).
           Y is a {mat.shape} matrix with singular values of M as diagonal entries.
""")
    print(f"Shape of X -> {x.shape}, Shape of Y -> {y.shape}, Shape of Z -> {z.shape}")
    print(f"np.allclose(M, X @ Y @ Z) = {np.allclose(x @ np.r_[np.diag(y), np.zeros((4,2))] @ z, mat)}")
    print(np.array_str(x))
    print(f"Is X orthonormal (6 x 6) -> np.allclose(x.T @ x, np.eye(n)) = {np.allclose(x.T @ x, np.eye(n))}")
    x = x[:, :m]
    print(np.array_str(x))
    print(f"Is X orthonormal (6 x 2) : {np.allclose(x.T @ x, np.eye(m))}")

def SVDinWork(approx):
    im = Image.open('4.1.01.tiff').convert('L')
    # im.show()
    # print(im.format, im.size, im.mode)
    arr = np.asarray(im)              # dtype = np.uint8
    row, col = arr.shape
    maxRank = min((row, col))
    # print(arr)

    """ U, S, Vh = np.linalg.svd(arr, full_matrices=False)
    result = U @ np.diag(S) @ Vh
    print(np.allclose(arr, result))
    Image.fromarray(np.uint8(result), 'L').show()
    exit() """

    U, S, Vh = np.linalg.svd(arr, full_matrices=False)
    rank = S.shape[0]
    assert rank <= maxRank
    assert approx <= rank

    lowRankApprox = U[:, :approx] @ np.diag(S[:approx]) @ Vh[:approx, :]
    # print(lowRankApprox)

    newImg = Image.fromarray(np.uint8(lowRankApprox), 'L')
    newImg.show() # save('convert.jpeg')

# use local image via bytes
def imageToBytes(): 
    img = Image.open('4.1.01.tiff', mode='r')
    img_byte_arr = BytesIO()
    img.save(img_byte_arr, format='PNG')
    img_byte_arr = img_byte_arr.getvalue() 

# write on images 
def writeOnImages():
    fnt = ImageFont.truetype(r'calibri.ttf', 25)
    old_im = Image.open('4.1.01.jpg')
    originalSize = old_im.size
    newSize = tuple((9 * max(originalSize)) // 8 for _ in range(2))
    new_im = Image.new('L', newSize, "White")   ## luckily, this is already black!
    box = (
        (newSize[0] - originalSize[0]) // 2, 
        4 * (newSize[1] - originalSize[1]) // 5
    )
    new_im.paste(old_im, box)
    draw = ImageDraw.Draw(new_im)
    draw.text(
        (
            newSize[0] // 2, 
            (4 * box[1]) // 5
        ), 
        f"n = 5", 
        font = fnt, 
        anchor = "ms"
    ) # direction="ttb"
    new_im.show()

