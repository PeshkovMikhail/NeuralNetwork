from activations import *

class Flatten:
    def __init__(self):
        self.input_shape = None

    def forward(self, data: np.array):
        self.input_shape = data.shape
        batches = data.shape[0]
        res = data.reshape((batches, data.shape[1] * data.shape[2] * data.shape[3]))
        return res

    def __call__(self, x):
        return self.forward(x)
    
    def backward(self, err, lr):
        res = err.reshape(self.input_shape)
        return res

class ConvLayer:
    def __init__(self, c_in, c_out, kernel_size, stride = 1, padding = 0, activaton: ActivationFunc = Sigmoid()) -> None:
        self._weights = np.random.uniform(-1, 1, (c_out, c_in, kernel_size, kernel_size))
        self._c_in = c_in
        self._c_out = c_out
        self._kernel_size = kernel_size
        self._padding = padding
        self._stride = stride
        self.input = None
        self._bias = np.random.uniform(-1, 1, (c_out))
        self.act = activaton

    def forward(self, data: np.array):
        shape = np.array(data.shape)
        if len(shape) != 4 or shape[1] != self._c_in:
            raise ValueError("incorrect batch format")
        if shape[2] < self._kernel_size or shape[3] < self._kernel_size:
            raise ValueError("incorrent image size or kernel")
        
        self.input = data.copy()
        out_w, out_h = 1 + (shape[2:] + 2 * self._padding - self._kernel_size)//self._stride
        res = np.zeros((shape[0], self._c_out, out_w, out_h))

        for batch in range(shape[0]):
            for c_out in range(self._c_out):
                for c_in in range(self._c_in):
                    padded = np.pad(data[batch, c_in], 2)
                    for y in range(out_h):
                        for x in range(out_w):
                            sx = self._stride * x
                            sy = self._stride * y
                            crop = padded[sy: sy + self._kernel_size, sx : sx + self._kernel_size]
                            res[batch, c_out, y, x] += crop.reshape(self._kernel_size**2).dot(self._weights[c_out, c_in].reshape(self._kernel_size**2))
                res[batch, c_out] += self._bias[c_out]
        return self.act(res)
    
    def __call__(self, x):
        return self.forward(x)


    def backward(self, dout, lr):
        """
        A naive implementation of the backward pass for a convolutional layer.
        Inputs:
        - dout: Upstream derivatives.
        - cache: A tuple of (x, w, b, conv_param) as in conv_forward_naive
        Returns a tuple of:
        - dx: Gradient with respect to x
        - dw: Gradient with respect to w
        - db: Gradient with respect to b
        """
        
        # Initialisations
        dx = np.zeros_like(self.input)
        dw = np.zeros_like(self._weights)
        db = np.zeros_like(self._bias)
        
        # Dimensions
        N, C, H, W = self.input.shape
        F, _, HH, WW = self._weights.shape
        _, _, H_, W_ = dout.shape
        
        # db - dout (N, F, H', W')
        # On somme sur tous les éléments sauf les indices des filtres
        db = np.sum(dout, axis=(0, 2, 3))
        
        # dw = xp * dy
        # 0-padding juste sur les deux dernières dimensions de x
        xp = np.pad(self.input, ((0,), (0,), (self._padding,), (self._padding, )), 'constant')
        
        # Version sans vectorisation
        for n in range(N):       # On parcourt toutes les images
            for f in range(F):   # On parcourt tous les filtres
                for i in range(HH): # indices du résultat
                    for j in range(WW):
                        for k in range(H_): # indices du filtre
                            for l in range(W_):
                                for c in range(C): # profondeur
                                    dw[f,c,i,j] += xp[n, c, self._stride*i+k, self._stride*j+l] * dout[n, f, k, l]

        # dx = dy_0 * w'
        # Valide seulement pour un stride = 1
        # 0-padding juste sur les deux dernières dimensions de dy = dout (N, F, H', W')
        doutp = np.pad(dout, ((0,), (0,), (WW-1,), (HH-1, )), 'constant')

        # 0-padding juste sur les deux dernières dimensions de dx
        dxp = np.pad(dx, ((0,), (0,), (self._padding,), (self._padding, )), 'constant')

        # filtre inversé dimension (F, C, HH, WW)
        w_ = np.zeros_like(self._weights)
        for i in range(HH):
            for j in range(WW):
                w_[:,:,i,j] = self._weights[:,:,HH-i-1,WW-j-1]
        
        # Version sans vectorisation
        for n in range(N):       # On parcourt toutes les images
            for f in range(F):   # On parcourt tous les filtres
                for i in range(H+2*self._padding): # indices de l'entrée participant au résultat
                    for j in range(W+2*self._padding):
                        for k in range(HH): # indices du filtre
                            for l in range(WW):
                                for c in range(C): # profondeur
                                    dxp[n,c,i,j] += doutp[n, f, i+k, j+l] * w_[f, c, k, l]
        #Remove padding for dx
        dx = dxp[:,:,self._padding:-self._padding,self._padding:-self._padding]

        self._weights -= dw * lr
        self._bias -= db * lr
        return dx