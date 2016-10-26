require 'hdf5'
attenmaps=torch.Tensor(100,196)
imgidx=torch.Tensor(100)
myFile = hdf5.open('AttenmapsAndImgidx.h5', 'w')
myFile:write('/attenmaps', attenmaps)
myFile:write('/imgidx', imgidx)
myFile:close()


myFile = hdf5.open('AttenmapsAndImgidx.h5', 'r')
data = myFile:read('imgidx'):all()
myFile:close()
print(#data)
