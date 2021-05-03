import numpy as np
from gdatav4 import gdatav4
from scipy.io import loadmat


def test_gdatav4(plot=False):
    np.random.seed(1337)
    A = np.random.random((5,3))
    A = A.T
    h=3;
    w=5;
    x = np.linspace(1,10,w);
    y = np.linspace(1,10,h);
    X,Y = np.meshgrid(x,y);
    xq = np.linspace(1,10,w*10);
    yq = np.linspace(1,10,h*10);
    XQ,YQ=np.meshgrid(xq,yq);
    _,_,vq = gdatav4(X,Y,A,XQ,YQ)
    matlab = loadmat(r'tests\gdatav4_1337.mat')['vq']
    assert np.all(np.isclose(matlab,vq))
    if plot:
        import matplotlib.pyplot as plt
        fig,axes = plt.subplots(1,2)
        axes[0].imshow(vq)
        axes[0].set_title('Python')
        axes[1].imshow(matlab)
        axes[1].set_title('Matlab')
        plt.show()
