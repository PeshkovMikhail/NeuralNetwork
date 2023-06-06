from activations import *
from layer import *

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

class ConvLayer(Layer):
    def __init__(self, c_in, c_out, kernel_size, stride = 1, padding = 0, activaton: ActivationFunc = Sigmoid()) -> None:
        super().__init__()
        self._weights = np.random.uniform(-1, 1, (c_out, c_in, kernel_size, kernel_size))
        self.grad_w = np.zeros_like(self._weights)
        self._c_in = c_in
        self._c_out = c_out
        self._kernel_size = kernel_size
        self._padding = padding
        self._stride = stride
        self.input = None
        self._bias = np.random.uniform(-1, 1, (c_out))
        self.grad_b = np.zeros_like(self._bias)
        self.act = activaton
        self.res = None

    def forward(self, data: np.array):
        shape = np.array(data.shape)
        if len(shape) != 4 or shape[1] != self._c_in:
            raise ValueError("incorrect batch format")
        if shape[2] < self._kernel_size or shape[3] < self._kernel_size:
            raise ValueError("incorrent image size or kernel")
        
        out_w, out_h = 1 + (shape[2:] + 2 * self._padding - self._kernel_size)//self._stride
        self.res = np.zeros((shape[0], self._c_out, out_w, out_h))

        self.input = padded = np.pad(data, ((0,), (0,), (self._padding,), (self._padding, )), 'constant')
        for y in range(out_h):
            for x in range(out_w):
                sx = self._stride * x
                sy = self._stride * y
                self.res[:, :, y, x] = np.sum(self._weights[:,:,:,:]*padded[:, :, sy: sy+self._kernel_size, sx: sx + self._kernel_size],axis=(1, 2,3,))+ self._bias[np.newaxis,:]
        if self.act:
            self.res = self.act(self.res)
        return self.res
    
    def __call__(self, x):
        return self.forward(x)


    def backward(self, err, lr):
        if self.act:
            err = err * self.act.derivative(self.res)

        _, _, H, W = err.shape
        din = np.zeros_like(self.input)
        self.grad_w = np.zeros_like(self._weights)
        for y in range(H):
            for x in range(W):
                sy = y * self._stride
                sx = x * self._stride
                syk = sy + self._kernel_size
                sxk = sx + self._kernel_size
                e = err[:,:,np.newaxis,np.newaxis,np.newaxis,y,x]
                r = self.input[:,np.newaxis,:, sy:syk, sx:sxk]*e
                self.grad_w += np.sum(r, axis=0)

                din[:,np.newaxis,:, sy:syk, sx:sxk] += np.sum(self._weights[np.newaxis,:,:]*e,
                                                              axis = 1,keepdims=True)

        
        self.grad_b = np.sum(err,axis=(0,2,3))  
        self._weights -= lr * self.grad_w
        self._bias -= lr*self.grad_b
        return din[:,:,self._padding:-self._padding,self._padding:-self._padding]
    
class MaxPool:
    def __init__(self, scale) -> None:
        self.scale = scale
        self.pos = None

    def forward(self, data):
        s = data.shape
        res = np.zeros((s[0], s[1], s[2]//self.scale, s[3]//self.scale))
        self.pos = np.zeros((s[0], s[1], s[2]//self.scale, s[3]//self.scale, 2))
        for n in range(s[0]):
            for c in range(s[1]):
                for y in range(s[2]//self.scale):
                    for x in range(s[3]//self.scale):
                        sy = y*self.scale
                        sx = x*self.scale
                        crop = data[n, c, sy:sy+2, sx:sx+2]
                        i_t, j_t = np.where(np.max(crop) == crop)
                        i_t, j_t = i_t[0], j_t[0]
                        res[n, c, y, x] = np.max(crop)
                        self.pos[n, c, y, x] = [i_t, j_t]
        return res
    
    def __call__(self, x):
        return self.forward(x)
    
    def backward(self, err, lr):
        N, C, h, w = err.shape
        res = np.zeros((err.shape[0], err.shape[1], h*self.scale, w*self.scale))
        for n in range(N):
            for c in range(C):
                for y in range(h):
                    for x in range(w):
                        sub_y, sub_x = self.pos[n,c,y,x]
                        sub_y, sub_x = int(sub_y), int(sub_x)
                        sy = int(y*self.scale)
                        sx = int(x*self.scale)
                        e = err[n, c, y, x]
                        res[n, c, sy+sub_y, sx+sub_x] = e
        return res
                