import numpy as np
import cv2


def norm_image(img):
    """
    Normalize PIL image

    Normalizes luminance to (mean,std)=(0,1), and applies a [1%, 99%] contrast stretch
    """
    img_y, img_b, img_r = img.convert('YCbCr').split()

    img_y_np = np.asarray(img_y).astype(float)

    img_y_np /= 255
    img_y_np -= img_y_np.mean()
    img_y_np /= img_y_np.std()
    scale = np.max([np.abs(np.percentile(img_y_np, 1.0)),
                    np.abs(np.percentile(img_y_np, 99.0))])
    img_y_np = img_y_np / scale
    img_y_np = np.clip(img_y_np, -1.0, 1.0)
    img_y_np = (img_y_np + 1.0) / 2.0

    img_y_np = (img_y_np * 255 + 0.5).astype(np.uint8)

    img_y = Image.fromarray(img_y_np)

    img_ybr = Image.merge('YCbCr', (img_y, img_b, img_r))

    img_nrm = img_ybr.convert('RGB')

    return img_nrm

def peer2(img):
    b, g, r = cv2.split(img)

    ret, m1 = cv2.threshold(r, 220, 255, cv2.THRESH_BINARY)
    ret, m2 = cv2.threshold(g, 210, 255, cv2.THRESH_BINARY)
    ret, m3 = cv2.threshold(b, 170, 255, cv2.THRESH_BINARY)
    ret, m4 = cv2.threshold(cv2.absdiff(r, g), 15, 255, cv2.THRESH_BINARY)
    m5 = cv2.compare(r, g, cv2.CMP_GT)
    m6 = cv2.compare(r, b, cv2.CMP_GE)

    mask = m1 & m2 & m3 & m4 & m5 & m6

    return mask


def peer(img):
    b, g, r = cv2.split(img)
    ret, m1 = cv2.threshold(r, 95, 255, cv2.THRESH_BINARY)
    ret, m2 = cv2.threshold(g, 30, 255, cv2.THRESH_BINARY)
    ret, m3 = cv2.threshold(b, 20, 255, cv2.THRESH_BINARY)
    mmax = cv2.max(r, cv2.max(g, b))
    mmin = cv2.min(r, cv2.min(g, b))

    ret, m4 = cv2.threshold(mmax - mmin, 15, 255, cv2.THRESH_BINARY)
    ret, m5 = cv2.threshold(cv2.absdiff(r, g), 15, 255, cv2.THRESH_BINARY)
    m6 = cv2.compare(r, g, cv2.CMP_GE)
    m7 = cv2.compare(r, b, cv2.CMP_GE)
    mask = m1 & m2 & m3 & m6 & m4 & m5 & m7
    cv2.imshow("b", b)
    cv2.imshow("g", g)
    cv2.imshow("r", r)
    cv2.imshow('r_thre', m1)
    cv2.imshow('g_thre',m2)
    cv2.imshow('b_thre',m3)
    cv2.imshow('max-min',m4)
    cv2.imshow('absdiff',m5)
    cv2.imshow('r_g',m6)
    cv2.imshow('r_b',m7)
    cv2.imshow('res',mask)
    return mask

def peer_cost(img):
    b,g,r = cv2.split(img)

    ret, m1 = cv2.threshold(r, 95, 255, cv2.THRESH_BINARY)
    ret, m2 = cv2.threshold(g, 40, 255, cv2.THRESH_BINARY)
    ret, m3 = cv2.threshold(b, 20, 255, cv2.THRESH_BINARY)

    mmax = cv2.max(r, cv2.max(g, b))
    mmin = 255
    for tmp in b[0]:
        if (tmp != 0 and tmp < mmin):
            mmin = tmp
    for tmp in g[0]:
        if (tmp != 0 and tmp < mmin):
            mmin = tmp
    for tmp in r[0]:
        if (tmp != 0 and tmp < mmin):
            mmin = tmp
    print mmin
    ret, m4 = cv2.threshold(mmax - mmin, 15, 255, cv2.THRESH_BINARY)
    ret, m5 = cv2.threshold(cv2.absdiff(r, g), 15, 255, cv2.THRESH_BINARY)
    m6 = cv2.compare(r, g, cv2.CMP_GT)
    m7 = cv2.compare(r, b, cv2.CMP_GE)

    mask = m1 & m2 & m3 & m4 & m5 & m6 & m7

    return mask

def ndrplz(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(img)
    ret, h1m = cv2.threshold(h, 5, 255, cv2.THRESH_BINARY)
    ret, h2m = cv2.threshold(h, 17, 255, cv2.THRESH_BINARY_INV)
    ret, s1m = cv2.threshold(s, 38, 255, cv2.THRESH_BINARY)
    ret, s2m = cv2.threshold(s, 250, 255, cv2.THRESH_BINARY_INV)
    ret, v1m = cv2.threshold(v, 51, 255, cv2.THRESH_BINARY)
    ret, v2m = cv2.threshold(v, 242, 255, cv2.THRESH_BINARY_INV)

    mask = h1m & h2m & s1m & s2m & v1m & v2m

    return mask

def mask_black_bkgd(img):
    #Invert the image to be white on black for compatibility with findContours function.

    imgray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    #Binarize the image and call it thresh.
    ret, thresh = cv2.threshold(imgray, 127, 255, cv2.THRESH_BINARY)

    #Find all the contours in thresh. In your case the 3 and the additional strike
    im2, contours, hierarchy = cv2.findContours(thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    #Calculate bounding rectangles for each contour.
    rects = [cv2.boundingRect(cnt) for cnt in contours]

    #Calculate the combined bounding rectangle points.
    top_x = min([x for (x, y, w, h) in rects])
    top_y = min([y for (x, y, w, h) in rects])
    bottom_x = max([x+w for (x, y, w, h) in rects])
    bottom_y = max([y+h for (x, y, w, h) in rects])

    #Draw the rectangle on the image
    #out = cv2.rectangle(img, (top_x, top_y), (bottom_x, bottom_y), (0, 255, 0), 2)
    crop = img[top_y:bottom_y,top_x:bottom_x]
    return crop #thresh

def ahlbert(img):
    # img = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2YCR_CB)
    # img.convertTo(img, cv2.CV_32FC3)
    # img = np.float32(img)
    y, cb, cr = cv2.split(img)
    ret, cr1m = cv2.threshold(cr, 138, 255, cv2.THRESH_BINARY)
    ret, cr2m = cv2.threshold(cr, 178, 255, cv2.THRESH_BINARY_INV)
    cr = np.float32(cr)
    cb = np.float32(cb)
    sum = cb + 0.6 * cr
    ret, sum1m = cv2.threshold(sum, 200, 255, cv2.THRESH_BINARY)
    ret, sum2m = cv2.threshold(sum, 215, 255, cv2.THRESH_BINARY_INV)
    sum1m = np.uint8(sum1m)
    sum2m = np.uint8(sum2m)


def sobottka(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(img)
    ret, h1m = cv2.threshold(h, 0, 400, cv2.THRESH_BINARY)
    ret, h2m = cv2.threshold(h, 35, 400, cv2.THRESH_BINARY_INV)
    ret, s1m = cv2.threshold(s, 58, 400, cv2.THRESH_BINARY)
    ret, s2m = cv2.threshold(s, 173, 400, cv2.THRESH_BINARY_INV)
    mask = h1m & h2m & s1m & s2m
    return mask


import numpy as np
import cv2
from matplotlib import pyplot as plt

def add_to_fig(img,plt):
    hist, bins = np.histogram(img.flatten(), 256, [0, 256])
    cdf = hist.cumsum()
    cdf_normalized = cdf * hist.max() / cdf.max()  # this line not necessary.
    fig.add_subplot(cdf_normalized)

def plt_img(img,plt):
    hist, bins = np.histogram(img.flatten(), 256, [1, 256])

    cdf = hist.cumsum()
    cdf_normalized = cdf * hist.max() / cdf.max()  # this line not necessary.

    plt.plot(cdf_normalized, color='b')
    plt.hist(img.flatten(), 256, [1, 256], color='r')
    plt.xlim([1, 256])
    plt.legend(('cdf', 'histogram'), loc='upper left')
    #plt.show()

import sympy as sm

def derivate(img):
    hist, bins = np.histogram(img.flatten(), 256, [1, 256])

    cdf = hist.cumsum()
    cdf_normalized = cdf * hist.max() / cdf.max()
    print cdf[80:100]
    print cdf.shape
   # print sm.diff(cdf,np.arange(0,256))
    #print np.gradient(cdf[80:100])
    res = 0
    for i in xrange(0,len(cdf)) :
        if (cdf[i] > 2000):
            res = i
            break
    print res
    plt.plot(np.gradient(cdf))
    return res
    # Capture frame-by-frame
    #frame = cv2.imread("/home/davinci/Downloads/photo_2017-11-18_14-32-53.jpg",1)
#img = cv2.imread('/home/davinci/Downloads/photo_2017-11-18_14-32-53.jpg', 1)
#img = cv2.imread("/home/davinci/Downloads/photo_2017-11-18_16-04-31.jpg", 1)
fig = plt.figure()
img = cv2.imread("/home/davinci/Downloads/E_vAXcQF7Pg.jpg", 1)
plt.figure(1)
plt.subplot(311)

#plt_img(img)
plt_img(img, plt)

hist, bins = np.histogram(img.flatten(), 256, [0, 256])
fig.add_subplot()
cdf = hist.cumsum()

cdf_m = np.ma.masked_equal(cdf,0)
cdf_m = (cdf_m - cdf_m.min())*255/(cdf_m.max()-cdf_m.min())
cdf = np.ma.filled(cdf_m,0).astype('uint8')

img2 = cdf[img]
#median = cv2.medianBlur(img2,55)
res = peer(img2)
#cv2.imshow("first",img)

#plt_img(img2)
plt.subplot(312)

plt_img(img2,plt)
#tmp = cv2.bitwise_and(img, img, mask=res)
res_peer = cv2.bitwise_and(img2,img2, mask = res)
plt.subplot(313)
plt_img(res_peer,plt)
gray = cv2.cvtColor(res_peer,cv2.COLOR_RGB2GRAY)
ret,mask = cv2.threshold(gray,derivate(res_peer),255,cv2.THRESH_BINARY)

last_res = cv2.bitwise_and(img2,img2, mask = mask)
cv2.imshow("second",last_res)
cv2.imshow("normalize",img2)
cv2.waitKey(200)
plt.show()
cv2.waitKey()

cv2.destroyAllWindows()
