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
        self.res = None

    def forward(self, data: np.array):
        shape = np.array(data.shape)
        if len(shape) != 4 or shape[1] != self._c_in:
            raise ValueError("incorrect batch format")
        if shape[2] < self._kernel_size or shape[3] < self._kernel_size:
            raise ValueError("incorrent image size or kernel")
        
        self.input = data.copy()
        out_w, out_h = 1 + (shape[2:] + 2 * self._padding - self._kernel_size)//self._stride
        self.res = np.zeros((shape[0], self._c_out, out_w, out_h))

        for batch in range(shape[0]):
            for c_out in range(self._c_out):
                for c_in in range(self._c_in):
                    padded = np.pad(data[batch, c_in], 2)
                    for y in range(out_h):
                        for x in range(out_w):
                            sx = self._stride * x
                            sy = self._stride * y
                            crop = padded[sy: sy + self._kernel_size, sx : sx + self._kernel_size]
                            self.res[batch, c_out, y, x] += crop.reshape(self._kernel_size**2).dot(self._weights[c_out, c_in].reshape(self._kernel_size**2))
                self.res[batch, c_out] += self._bias[c_out]
        self.res = self.act(self.res)
        return self.res
    
    def __call__(self, x):
        return self.forward(x)


    def backward(self, dout, lr):
        dout = dout* self.act.derivative(self.res)

        dx = np.zeros_like(self.input)
        dw = np.zeros_like(self._weights)
        db = np.zeros_like(self._bias)
        
        N, C, H, W = self.input.shape
        _, _, H_, W_ = dout.shape

        db = np.sum(dout, axis=(0, 2, 3))
        
        xp = np.pad(self.input, ((0,), (0,), (self._padding,), (self._padding, )), 'constant')
        
        for n in range(N):
            for f in range(self._c_out):
                for i in range(self._kernel_size):
                    for j in range(self._kernel_size):
                        for c in range(C):
                            si = self._stride*i
                            sj = self._stride*j
                            dw[f,c,i,j] += np.sum(xp[n,c,si:si+H_, sj: sj + W_]* dout[n,f,:,:]) 

        doutp = np.pad(dout, ((0,), (0,), (self._kernel_size-1,), (self._kernel_size-1, )), 'constant')

        dxp = np.pad(dx, ((0,), (0,), (self._padding,), (self._padding, )), 'constant')

        #w_ = np.zeros_like(self._weights)
        w_ = self._weights.copy()#[:, :, ::-1, ::-1]
        
        for n in range(N):
            for f in range(self._c_out):
                for i in range(H+2*self._padding):
                    for j in range(W+2*self._padding):
                        for c in range(C):
                            dxp[n,c, i, j] = np.sum(doutp[n,f, i:i+self._kernel_size, j: j + self._kernel_size]*w_[f, c])

        dx = dxp[:,:,self._padding:-self._padding,self._padding:-self._padding]

        self._weights -= dw * lr
        self._bias -= db * lr
        return dx
    
class MaxPool:
    def __init__(self, scale) -> None:
        self.scale = scale

    def forward(self, data):
        s = data.shape
        res = np.zeros((s[0], s[1], s[2]//self.scale, s[3]//self.scale))
        for y in range(s[2]//self.scale):
            for x in range(s[3]//self.scale):
                sy = y*self.scale
                sx = x*self.scale
                res[:, :, y, x] = np.max(data[:, :, sy:sy+self.scale, sx:sx+self.scale], axis=(2,3))
        return res
    
    def __call__(self, x):
        return self.forward(x)
    
    def backward(self, err, lr):
        h, w = err.shape[2:]
        h, w = h*self.scale, w*self.scale
        d = self.scale**2
        res = np.zeros((err.shape[0], err.shape[1], h, w))
        for y in range(h):
            for x in range(w):
                res[:, :, y, x] = err[:, :, y//self.scale, x//self.scale]/d
        return res
                