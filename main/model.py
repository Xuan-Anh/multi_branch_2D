import torch 
import torch.nn as nn 
class DS_Block(nn.Module):
    def __init__(self, in_filters, out_filters):
        super().__init__()
        self.bn1 = nn.BatchNorm2d(in_filters)
        self.conv1 = nn.Conv2d(in_filters, out_filters, kernel_size = 3)
        self.bn2 = nn.BatchNorm2d(out_filters)
        self.conv2 = nn.Conv2d(out_filters, out_filters, kernel_size = 3)
        self.maxpool = nn.MaxPool2d(2)
                
    def forward(self, x):
        x = self.bn1(x)
        x = self.conv1(x)
        x = self.bn2(x)
        x = self.conv2(x)
        x_pool = self.maxpool(x)
        x_skip = x
        print('[INFO] DS_Block: ')
        print('------ x_skip: ', x_skip.shape)
        print('------ x_pool: ', x_pool.shape)
        return x_skip, x_pool

class Bottom_Block(nn.Module):
    def __init__(self, in_filters, out_filters):
        super().__init__()
        self.bn1 = nn.BatchNorm2d(in_filters)
        self.conv1 = nn.Conv2d(in_filters, out_filters, kernel_size = 3)
        self.bn2 = nn.BatchNorm2d(out_filters)
        self.conv2 = nn.Conv2d(out_filters, out_filters, kernel_size = 3)
                
    def forward(self, x):
        x = self.bn1(x)
        x = self.conv1(x)
        x = self.bn2(x)
        x = self.conv2(x)
        return x
    
class MMFF_Block(nn.Module):
    '''Multi-Modality Feature Fusion block'''
    def __init__(self, in_filters, out_filters):
        super().__init__()
       
        self.bn1_1 = nn.BatchNorm2d(in_filters)        
        self.bn1_2 = nn.BatchNorm2d(in_filters)
        self.bn1_3 = nn.BatchNorm2d(in_filters)
        # 1x1 conv
        self.conv1_1 = nn.Conv2d(in_filters, out_filters, kernel_size= 1)
        self.conv1_2 = nn.Conv2d(in_filters, out_filters, kernel_size= 1)
        self.conv1_3 = nn.Conv2d(in_filters, out_filters, kernel_size= 1)
        
        # batch norm
        self.bn2_1 = nn.BatchNorm2d(out_filters)
        self.bn2_2 = nn.BatchNorm2d(out_filters)
        self.bn2_3 = nn.BatchNorm2d(out_filters)
        
        # 3x3 conv
        self.conv2_1 = nn.Conv2d(out_filters, out_filters, kernel_size= 3)
        self.conv2_2 = nn.Conv2d(out_filters, out_filters, kernel_size= 3)
        self.conv2_3 = nn.Conv2d(out_filters, out_filters, kernel_size= 3)
        
    def forward(self, x_skip_1, x_skip_2, x_skip_3):
        
        x_1 = self.bn1_1(x_skip_1)
        x_1 = self.conv1_1(x_1)
        x_1 = self.bn2_1(x_1)
        x_1 = self.conv2_1(x_1)
        
        x_2 = self.bn1_2(x_skip_2)
        x_2 = self.conv1_2(x_2)
        x_2 = self.bn2_2(x_2)
        x_2 = self.conv2_2(x_2)
        
        x_3 = self.bn1_3(x_skip_3)
        x_3 = self.conv1_3(x_3)
        x_3 = self.bn2_3(x_3)
        x_3 = self.conv2_3(x_3)
        
        # concat all x_1, x_2, x_3
        x = torch.cat((x_1, x_2, x_3), dim = 1)
        return x

counter = 0 
class MSFU_Block(nn.Module):
    '''
    Multi-Scale Feature Up-sampling block
    '''
    def __init__(self, in_filters, out_filters):
        super().__init__()
        # in_filters = in_filters * 3
        # out_filters = out_filters * 3
        print('[INFO] MSFU-Block:------------ ')
        print('------ in_filters: ', in_filters)
        print('------ out_filters: ', out_filters)
        
        
        self.bn1 = nn.BatchNorm2d(in_filters)
        self.conv1 = nn.Conv2d(in_filters, out_filters, kernel_size = 1)
        # 2x2 upsample
        # self.upsample = nn.Upsample(scale_factor = 2, mode = 'bilinear', align_corners = True)
        # 2x2 conv2Dtranspose
        self.upsample = nn.ConvTranspose2d(out_filters, out_filters, kernel_size = 2, stride = 2)
        self.sigmoid = nn.Sigmoid()
        # CONCATENATE
        # after concate
        self.bn2 = nn.BatchNorm2d(out_filters*2)        
        self.conv2 = nn.Conv2d(out_filters*2, out_filters, kernel_size = 1)
        self.bn3 = nn.BatchNorm2d(out_filters)
        self.conv3 = nn.Conv2d(out_filters, out_filters, kernel_size = 3)

    def forward(self, x_low, x_high):
        ''' 
        low-resolution input to the MSFU-Block came from the bottom- or another MSFU-Block
        high-resolution came from MMFF-Block
        '''
        print('$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$')
        print('[SHAPE] x_low  ==: ', x_low.shape)
        print('[SHAPE] x_high == ', x_high.shape)
        global counter
        counter += 1
        print('[COUNTER]  counter  = ', counter)
        x_low = self.bn1(x_low)
        x_low = self.conv1(x_low)
        x_low = self.sigmoid(x_low)
        x_low = self.upsample(x_low)
        
        print('[SHAPE] x_low after: ', x_low.shape)
        x = torch.cat((x_low, x_high), dim = 1)
        x = self.bn2(x)
        x = self.conv2(x)
        x = self.bn3(x)
        x = self.conv3(x)
        return x

class MultiBranch2DNet(nn.Module):
    def __init__(self, filter_list_in = [1, 32, 64, 128, 256, 512], filter_list_out = [1, 32*3, 64*3, 128*3, 256*3, 512*3]):
        # filter_list_in = [3, 32, 64, 128, 256, 512]
        # filter_list_out = [1, 32, 64, 128, 256, 512]
        # len(filter_list_in) == len(filter_list_out)
        super().__init__()
        self.DS_list_1 = [DS_Block(filter_list_in[i], filter_list_in[i+1]) for i in range(len(filter_list_in)-2)]
        self.DS_list_2 = [DS_Block(filter_list_in[i], filter_list_in[i+1]) for i in range(len(filter_list_in)-2)]
        self.DS_list_3 = [DS_Block(filter_list_in[i], filter_list_in[i+1]) for i in range(len(filter_list_in)-2)]
        
        self.BTM_1 = Bottom_Block(filter_list_in[-2], filter_list_in[-1])
        self.BTM_2 = Bottom_Block(filter_list_in[-2], filter_list_in[-1])
        self.BTM_3 = Bottom_Block(filter_list_in[-2], filter_list_in[-1])
        
        self.MMFF_list = [MMFF_Block(filter_list_in[i+1], filter_list_in[i]) for i in range(len(filter_list_in)-1)]
        self.MSFU_list = [MSFU_Block(filter_list_out[i+1], filter_list_out[i]) for i in range(len(filter_list_out)-2)]
        self.len_ds_list = len(self.DS_list_1 )
    def forward(self, x_1, x_2, x_3):
        # use setattr to create new variables
        # init input
        print('[CHECK] x_1.shape', x_1.shape)
        setattr(self, 'x_ds_1_0_pool', x_1)
        setattr(self, 'x_ds_2_0_pool', x_2)
        setattr(self, 'x_ds_3_0_pool', x_3)
        
        # pass through DS blocks                
        for i in [1, 2, 3]:
            for j in range(self.len_ds_list):
                output_ds = self.DS_list_1[j](getattr(self, f'x_ds_{i}_{j}_pool'))
                setattr(self, f'x_ds_{i}_{j+1}_skip', output_ds[0])
                setattr(self, f'x_ds_{i}_{j+1}_pool', output_ds[1])
        print('[DONE] DS blocks') 

        x_btm_1 = self.BTM_1(getattr(self, 'x_ds_1_'+str(self.len_ds_list) + '_pool'))
        x_btm_2 = self.BTM_2(getattr(self, 'x_ds_2_'+str(self.len_ds_list) + '_pool'))
        x_btm_3 = self.BTM_3(getattr(self, 'x_ds_3_'+str(self.len_ds_list) + '_pool'))
        print('[DONE] BTM blocks')        
        
        setattr(self, 'x_mmff_5', self.MMFF_list[self.len_ds_list](x_btm_1, x_btm_2, x_btm_3))
        for i in range(self.len_ds_list):
            setattr(self, 'x_mmff_' + str(i+1), self.MMFF_list[i](
                                    getattr(self, 'x_ds_1_'+str(i+1) + '_skip'), \
                                    getattr(self, 'x_ds_2_'+str(i+1) + '_skip'), \
                                    getattr(self, 'x_ds_3_'+str(i+1) + '_skip')))
        print('[DONE] MMFF blocks')
        
        print('len_ds_list = ', self.len_ds_list)
        print('x_mmff_5.shape = ', getattr(self, 'x_mmff_5').shape)
        print('x_mmff_4.shape = ', getattr(self, 'x_mmff_4').shape)
        print('x_mmff_3.shape = ', getattr(self, 'x_mmff_3').shape)
        print('x_mmff_2.shape = ', getattr(self, 'x_mmff_2').shape)
        print('x_mmff_1.shape = ', getattr(self, 'x_mmff_1').shape)
        # print('x_mmff_0.shape = ', getattr(self, 'x_mmff_0').shape)
        
        setattr(self, 'x_msfu_4' ,  self.MSFU_list[self.len_ds_list - 1](getattr(self, 'x_mmff_5'), getattr(self, 'x_mmff_4')))
        print('[SHAPE] x_msfu_4.shape = ', getattr(self, 'x_msfu_4').shape)
        
        for i in reversed(range(self.len_ds_list - 1)):
            print('[i] = ', i)
            setattr(self, 'x_msfu_' + str(i+1), self.MSFU_list[i]( getattr(self, 'x_msfu_' + str(i+2)), getattr(self, 'x_mmff_' + str(i+1))))
        print('[DONE] MSFU blocks')
        
        x = getattr(self, 'x_msfu_1')
        return x 

IMG_SIZE = 256
x_1 = torch.randn(12, 1, IMG_SIZE, IMG_SIZE)
x_2 = torch.randn(12, 1, IMG_SIZE, IMG_SIZE)
x_3 = torch.randn(12, 1, IMG_SIZE, IMG_SIZE)

model_branch_2d = MultiBranch2DNet()
model_branch_2d(x_1, x_2, x_3)

# x_mmff_5.shape =  torch.Size([12, 768, 116, 116])
# x_mmff_4.shape =  torch.Size([12, 384, 240, 240])
# x_mmff_3.shape =  torch.Size([12, 192, 244, 244])
# x_mmff_2.shape =  torch.Size([12, 96, 248, 248])
# x_mmff_1.shape =  torch.Size([12, 3, 252, 252])
# msfu_ok = MSFU_Block(384, 192)
# X = torch.randn(12, 384, 240, 240)
# Y = torch.randn(12, 192, 244, 244)
# msfu_ok(Y, X).shape