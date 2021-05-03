import numpy as np

def mergesimpts(data,tols,mode='average'):
    data_ = data.copy()[np.argsort(data[:,0])]
    newdata = []
    tols_ = np.array(tols)
    idxs_ready =[]
    point = 0
    for point in range(data_.shape[0]):
        if point in idxs_ready:
            continue
        else:
            similar_pts = np.where(np.prod(np.abs(data_ - data_[point]) < tols_, axis=-1))
            similar_pts = np.array(list(set(similar_pts[0].tolist())- set(idxs_ready)))
            idxs_ready += similar_pts.tolist()
            if mode == 'average':
                exemplar = np.mean(data_[similar_pts],axis=0)
            else:
                exemplar = data_[similar_pts].copy()[0] # first
            newdata.append(exemplar)
    return np.array(newdata)

def mergepoints2D(x,y,v):
    # Sort x and y so duplicate points can be averaged

    # Need x,y and z to be column vectors

    sz = x.size
    x = x.copy()
    y = y.copy()
    v = v.copy()
    x = np.reshape(x,(sz),order='F');
    y = np.reshape(y,(sz),order='F');
    v = np.reshape(v,(sz),order='F');

    myepsx = np.spacing(0.5 * (np.max(x) - np.min(x)))**(1/3);
    myepsy = np.spacing(0.5 * (np.max(y) - np.min(y)))**(1/3);
    # % look for x, y points that are indentical (within a tolerance)
    # % average out the values for these points
    if np.all(np.isreal(v)):
        data = np.stack((y,x,v), axis=-1)
        yxv = mergesimpts(data,[myepsy,myepsx,np.inf],'average')
        x = yxv[:,1]
        y = yxv[:,0]
        v = yxv[:,2]
    else:
        #% if z is imaginary split out the real and imaginary parts
        data = np.stack((y,x,np.real(v),np.imag(v)), axis=-1)
        yxv = mergesimpts(data,[myepsy,myepsx,np.inf,np.inf],'average')
        x = yxv[:,1]
        y = yxv[:,0]
        #% re-combine the real and imaginary parts
        v = yxv[:,2]+1j*yxv[:,3]
    #% give a warning if some of the points were duplicates (and averaged out)
    if sz > x.shape[0]:
        print('MATLAB:griddata:DuplicateDataPoints')
    return x,y,v

def gdatav4(x,y,v,xq,yq):
    """
    %GDATAV4 MATLAB 4 GRIDDATA interpolation

    %   Reference:  David T. Sandwell, Biharmonic spline
    %   interpolation of GEOS-3 and SEASAT altimeter
    %   data, Geophysical Research Letters, 2, 139-142,
    %   1987.  Describes interpolation using value or
    %   gradient of value in any dimension.
    """
    x, y, v = mergepoints2D(x,y,v);

    xy = x + 1j*y
    xy = np.squeeze(xy)
    #% Determine distances between points
    
    # d = np.zeros((xy.shape[0],xy.shape[0]))
    # for i in range(xy.shape[0]):
    #     for j in range(xy.shape[0]):
    #         d[i,j]=np.abs(xy[i]-xy[j])

    d = np.abs(np.subtract.outer(xy, xy))
    # % Determine weights for interpolation
    g = np.square(d) * (np.log(d)-1) #% Green's function.
    # % Fixup value of Green's function along diagonal
    np.fill_diagonal(g, 0)
    weights = np.linalg.lstsq(g, v)[0]

    (m,n) = xq.shape
    vq = np.zeros(xq.shape);
    #xy = np.tranpose(xy);

    # % Evaluate at requested points (xq,yq).  Loop to save memory.
    for i in range(m):
        for j in range(n):
            d = np.abs(xq[i,j] + 1j*yq[i,j] - xy);
            g = np.square(d) * (np.log(d)-1);#   % Green's function.
            #% Value of Green's function at zero
            g[np.where(np.isclose(d,0))] = 0;
            vq[i,j] = (np.expand_dims(g,axis=0) @ np.expand_dims(weights,axis=1))[0][0]
    return xq,yq,vq
