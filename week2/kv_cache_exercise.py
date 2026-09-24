def append(self,k_new,v_new):
    if k_new.ndim!=4 or v_new.shape!=k_new.shape:
        raise ValueError("kv维度不对")
    b,h,count,d=k_new.shape
    if ((b,h,d)!=(self.k.size(0),self.k.size(1),self.k.size(3)) or count<=0):
        raise ValueError("KV参数不匹配")
    for tensor in (k_new, v_new):
        if tensor.device != self.k.device or tensor.dtype != self.k.dtype:
            raise ValueError("device/dtype 与缓存不匹配")
    end=self.length+count
    if end>self.capacity:
        raise ValueError("缓存容量不足")
    self.k[:,:,self.length:end,:].copy_(k_new)
    self.v[:,:,self.length:end,:].copy_(v_new)
    self.length=end

def view(self):
    return self.k[:,:,:self.length,:],self.v[:,:,:self.length,:]

def reset(self):
    self.length=0
