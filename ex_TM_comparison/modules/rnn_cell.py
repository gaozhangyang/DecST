from torch import nn
import torch
import torch.nn.functional as F
from torch.autograd import Variable
from torch.nn import Parameter
import math
import copy
from functools import reduce
import operator


# Cite from https://github.com/jhhuang96/ConvLSTM-PyTorch, Thanks! 
class ConvLSTM_cell(nn.Module):
    """ConvLSTMCell
    """
    def __init__(self, shape, input_channels, filter_size, num_features):
        super(ConvLSTM_cell, self).__init__()
        self.shape = shape  # H, W
        self.input_channels = input_channels
        self.filter_size = filter_size
        self.num_features = num_features
        # in this way the output has the same size
        self.padding = (filter_size - 1) // 2
        self.conv = nn.Sequential(
            nn.Conv2d(self.input_channels + self.num_features,
                      4 * self.num_features, self.filter_size, 1,
                      self.padding),
            nn.GroupNorm(4 * self.num_features // 32, 4 * self.num_features))

    def forward(self, inputs=None, hidden_state=None, seq_len=10):
        #  seq_len=10 for moving_mnist
        if hidden_state is None:
            hx = torch.zeros(inputs.size(1), self.num_features, self.shape[0],
                             self.shape[1]).cuda()
            cx = torch.zeros(inputs.size(1), self.num_features, self.shape[0],
                             self.shape[1]).cuda()
        else:
            hx, cx = hidden_state
        output_inner = []
        for index in range(seq_len):
            if inputs is None:
                x = torch.zeros(hx.size(0), self.input_channels, self.shape[0],
                                self.shape[1]).cuda()
            else:
                x = inputs[index, ...]

            combined = torch.cat((x, hx), 1)
            gates = self.conv(combined)  # gates: S, num_features*4, H, W
            # it should return 4 tensors: i,f,g,o
            ingate, forgetgate, cellgate, outgate = torch.split(
                gates, self.num_features, dim=1)
            ingate = torch.sigmoid(ingate)
            forgetgate = torch.sigmoid(forgetgate)
            cellgate = torch.tanh(cellgate)
            outgate = torch.sigmoid(outgate)

            cy = (forgetgate * cx) + (ingate * cellgate)
            hy = outgate * torch.tanh(cy)
            output_inner.append(hy)
            hx = hy
            cx = cy
        return torch.stack(output_inner), (hy, cy)


# Cite from https://github.com/thuml/predrnn-pytorch, Thanks!
class PredRNN_cell(nn.Module):
    def __init__(self, in_channel, num_hidden, width, filter_size, stride, layer_norm):
        super(PredRNN_cell, self).__init__()

        self.num_hidden = num_hidden
        self.padding = filter_size // 2
        self._forget_bias = 1.0
        if layer_norm:
            self.conv_x = nn.Sequential(
                nn.Conv2d(in_channel, num_hidden * 7, kernel_size=filter_size, stride=stride, padding=self.padding, bias=False),
                nn.LayerNorm([num_hidden * 7, width, width])
            )
            self.conv_h = nn.Sequential(
                nn.Conv2d(num_hidden, num_hidden * 4, kernel_size=filter_size, stride=stride, padding=self.padding, bias=False),
                nn.LayerNorm([num_hidden * 4, width, width])
            )
            self.conv_m = nn.Sequential(
                nn.Conv2d(num_hidden, num_hidden * 3, kernel_size=filter_size, stride=stride, padding=self.padding, bias=False),
                nn.LayerNorm([num_hidden * 3, width, width])
            )
            self.conv_o = nn.Sequential(
                nn.Conv2d(num_hidden * 2, num_hidden, kernel_size=filter_size, stride=stride, padding=self.padding, bias=False),
                nn.LayerNorm([num_hidden, width, width])
            )
        else:
            self.conv_x = nn.Sequential(
                nn.Conv2d(in_channel, num_hidden * 7, kernel_size=filter_size, stride=stride, padding=self.padding, bias=False),
            )
            self.conv_h = nn.Sequential(
                nn.Conv2d(num_hidden, num_hidden * 4, kernel_size=filter_size, stride=stride, padding=self.padding, bias=False),
            )
            self.conv_m = nn.Sequential(
                nn.Conv2d(num_hidden, num_hidden * 3, kernel_size=filter_size, stride=stride, padding=self.padding, bias=False),
            )
            self.conv_o = nn.Sequential(
                nn.Conv2d(num_hidden * 2, num_hidden, kernel_size=filter_size, stride=stride, padding=self.padding, bias=False),
            )
        self.conv_last = nn.Conv2d(num_hidden * 2, num_hidden, kernel_size=1, stride=1, padding=0, bias=False)

    def forward(self, x_t, h_t, c_t, m_t):
        x_concat = self.conv_x(x_t)
        h_concat = self.conv_h(h_t)
        m_concat = self.conv_m(m_t)
        i_x, f_x, g_x, i_x_prime, f_x_prime, g_x_prime, o_x = torch.split(x_concat, self.num_hidden, dim=1)
        i_h, f_h, g_h, o_h = torch.split(h_concat, self.num_hidden, dim=1)
        i_m, f_m, g_m = torch.split(m_concat, self.num_hidden, dim=1)

        i_t = torch.sigmoid(i_x + i_h)
        f_t = torch.sigmoid(f_x + f_h + self._forget_bias)
        g_t = torch.tanh(g_x + g_h)

        c_new = f_t * c_t + i_t * g_t

        i_t_prime = torch.sigmoid(i_x_prime + i_m)
        f_t_prime = torch.sigmoid(f_x_prime + f_m + self._forget_bias)
        g_t_prime = torch.tanh(g_x_prime + g_m)

        m_new = f_t_prime * m_t + i_t_prime * g_t_prime

        mem = torch.cat((c_new, m_new), 1)
        o_t = torch.sigmoid(o_x + o_h + self.conv_o(mem))
        h_new = o_t * torch.tanh(self.conv_last(mem))

        return h_new, c_new, m_new

# cite from https://github.com/coolsunxu/MIM_Pytorch, Thanks!
class MIMBlock(nn.Module):
	def __init__(self, layer_name, filter_size, num_hidden_in, num_hidden,
				 seq_shape, x_shape_in, tln=False, initializer=None):
		super(MIMBlock, self).__init__()
		
		"""Initialize the basic Conv LSTM cell.
		Args:
			layer_name: layer names for different convlstm layers.
			filter_size: int tuple thats the height and width of the filter.
			num_hidden: number of units in output tensor.
			forget_bias: float, The bias added to forget gates (see above).
			tln: whether to apply tensor layer normalization
		"""
		self.layer_name = layer_name # 当前网络层名
		self.filter_size = filter_size # 卷积核大小
		self.num_hidden_in = num_hidden_in # 隐藏层输入
		self.num_hidden = num_hidden # 隐藏层大小
		self.convlstm_c = None # 
		self.batch = seq_shape[0] # batch_size
		self.height = seq_shape[3] # 图片高度
		self.width = seq_shape[4] # 图片宽度
		self.seq_shape = seq_shape
		self.x_shape_in = x_shape_in # 通道数
		self.layer_norm = tln # 是否归一化
		self._forget_bias = 1.0 # 遗忘参数
			
		# MIMS
		
		# h_t
		self.mims_h_t = nn.Conv2d(self.num_hidden,
					self.num_hidden * 4,
					self.filter_size, 1, padding=2
					)
					
		# c_t
		self.ct_weight = nn.Parameter(torch.randn((self.num_hidden*2,self.height,self.width)))

		# x
		self.mims_x = nn.Conv2d(self.num_hidden,
					self.num_hidden * 4,
					self.filter_size, 1, padding=2
					)
					
		# oc
		self.oc_weight = nn.Parameter(torch.randn((self.num_hidden,self.height,self.width)))
					
		# bn
		self.bn_h_concat = nn.BatchNorm2d(self.num_hidden * 4)
		self.bn_x_concat = nn.BatchNorm2d(self.num_hidden * 4)
					
		# MIMBLOCK	
		# h
		self.t_cc = nn.Conv2d(self.num_hidden_in,
				self.num_hidden*3, # 网络输入 输出通道数
				self.filter_size, 1, padding=2 # 滤波器大小 步长 填充方式
				)
				
		# m
		self.s_cc = nn.Conv2d(self.num_hidden_in,
				self.num_hidden*4,  # 网络输入 输出通道数
				self.filter_size, 1, padding=2 # 滤波器大小 步长 填充方式
				)
				
		# x
		self.x_cc = nn.Conv2d(self.x_shape_in,
				self.num_hidden*4, # 网络输入 输出通道数
				self.filter_size, 1, padding=2 # 滤波器大小 步长 填充方式
				)
		
		# c 
		self.c_cc = nn.Conv2d(self.num_hidden*2,
				self.num_hidden,  # 网络输入 输出通道数
				1, 1, padding=0 # 滤波器大小 步长 填充方式
				)
		
		# bn
		self.bn_t_cc = nn.BatchNorm2d(self.num_hidden*3)
		self.bn_s_cc = nn.BatchNorm2d(self.num_hidden*4)
		self.bn_x_cc = nn.BatchNorm2d(self.num_hidden*4)
					
	def init_state(self): # 初始化lstm 隐藏层状态
		return torch.zeros((self.batch, self.num_hidden, self.height, self.width),
					dtype=torch.float32).cuda()

	def MIMS(self, x, h_t, c_t): # MIMS
		if h_t is None:
			h_t = self.init_state()
		if c_t is None:
			c_t = self.init_state()
			
		x = x[:self.batch]
		h_t = h_t[:self.batch]
		c_t = c_t[:self.batch]
		# h_t
		h_concat = self.mims_h_t(h_t)
		
		if self.layer_norm: 
			h_concat = self.bn_h_concat(h_concat)
		
		i_h, g_h, f_h, o_h = torch.split(h_concat, self.num_hidden, 1)

		# ct_weight
		# ct_activation = torch.mul(c_t.repeat([1,2,1,1]), self.ct_weight)
		# ct_activation = torch.mul(c_t.repeat([1,2,1,1]), torch.rand(128, 64, 64).cuda())

		# ct_activation = torch.mul(torch.rand(x.shape[0], self.num_hidden*2, self.height, self.width).cuda(), self.ct_weight)
		ct_activation = torch.mul(torch.cat([c_t,c_t],dim=1), self.ct_weight)
		i_c, f_c = torch.split(ct_activation, self.num_hidden, 1)

		i_ = i_h + i_c
		f_ = f_h + f_c
		g_ = g_h
		o_ = o_h

		if x is not None:
			# x 
			x_concat = self.mims_x(x)
			
			if self.layer_norm:
				x_concat = self.bn_x_concat(x_concat)
			i_x, g_x, f_x, o_x = torch.split(x_concat, self.num_hidden, 1)

			i_ = i_ +  i_x
			f_ = f_ +  f_x
			g_ = g_ +  g_x
			o_ = o_ +  o_x

		i_ = torch.sigmoid(i_)
		f_ = torch.sigmoid(f_ + self._forget_bias)
		c_new = f_ * c_t + i_ * torch.tanh(g_)

		# oc_weight
		o_c = torch.mul(c_new, self.oc_weight)
		# o_c = torch.mul(torch.rand(x.shape[0], self.num_hidden, self.height, self.width).cuda(), \
		# 				torch.rand(x.shape[0], self.num_hidden, self.height, self.width).cuda())
		h_new = torch.sigmoid(o_ + o_c) * torch.tanh(c_new)
		# h_new = torch.sigmoid(o_ + o_c) * torch.tanh(torch.rand_like(c_new))

		return h_new, c_new
		# return o_c, c_new

	def forward(self, x, diff_h, h, c, m, batch_size):
		self.batch = batch_size
		# 初始化隐藏层 记忆 空间
		if h is None:
			h = self.init_state()
		if c is None:
			c = self.init_state()
		if m is None:
			m = self.init_state()
		if diff_h is None:
			diff_h = torch.zeros_like(h)
			
		x = x[:self.batch]
		h = h[:self.batch]
		c = c[:self.batch]
		m = m[:self.batch]
		# h
		t_cc = self.t_cc(h)
		# m
		s_cc = self.s_cc(m)
		# x
		x_cc = self.x_cc(x)
			
		if self.layer_norm:
			t_cc = self.bn_t_cc(t_cc)
			s_cc = self.bn_s_cc(s_cc)
			x_cc = self.bn_x_cc(x_cc)

		i_s, g_s, f_s, o_s = torch.split(s_cc, self.num_hidden, 1)
		i_t, g_t, o_t = torch.split(t_cc, self.num_hidden, 1)
		i_x, g_x, f_x, o_x = torch.split(x_cc, self.num_hidden, 1)

		i = torch.sigmoid(i_x + i_t)
		i_ = torch.sigmoid(i_x + i_s)
		g = torch.tanh(g_x + g_t)
		g_ = torch.tanh(g_x + g_s)
		f_ = torch.sigmoid(f_x + f_s + self._forget_bias)
		o = torch.sigmoid(o_x + o_t + o_s)
		new_m = f_ * m + i_ * g_
		
		# MIMS
		c, convlstm_c = self.MIMS(diff_h, c, self.convlstm_c)
		
		self.convlstm_c = convlstm_c.clone().detach()
		new_c = c + i * g
		cell = torch.cat((new_c, new_m), 1)
		
		# c
		tmp_cell = self.c_cc(cell)				
		new_h = o * torch.tanh(tmp_cell)

		return new_h, new_c, new_m


class MIMN(nn.Module):
	def __init__(self, layer_name, filter_size, num_hidden, seq_shape, tln=True, initializer=0.001):
		super(MIMN, self).__init__()
		"""Initialize the basic Conv LSTM cell.
		Args:
			layer_name: layer names for different convlstm layers.
			filter_size: int tuple thats the height and width of the filter.
			num_hidden: number of units in output tensor.
			tln: whether to apply tensor layer normalization.
		"""
		self.layer_name = layer_name # 当前网络层名
		self.filter_size = filter_size # 卷积核大小
		self.num_hidden = num_hidden # 隐藏层大小
		self.layer_norm = tln # 是否归一化
		self.batch = seq_shape[0] # batch_size
		self.height = seq_shape[3] # 图片高度
		self.width = seq_shape[4] # 图片宽度
		self.seq_shape = seq_shape
		self._forget_bias = 1.0 # 遗忘参数
			
		# h_t
		self.h_t = nn.Conv2d(self.num_hidden,
					self.num_hidden * 4,
					self.filter_size, 1, padding=2
					)
					
		# c_t
		self.ct_weight = nn.Parameter(torch.randn((self.num_hidden*2,self.height,self.width)))

		# x
		self.x = nn.Conv2d(self.num_hidden,
					self.num_hidden * 4,
					self.filter_size, 1, padding=2
					)
					
		# oc
		self.oc_weight = nn.Parameter(torch.randn((self.num_hidden,self.height,self.width)))
					
		# bn 
		self.bn_h_concat = nn.BatchNorm2d(self.num_hidden * 4)
		self.bn_x_concat = nn.BatchNorm2d(self.num_hidden * 4)

	def init_state(self): # 初始化lstm 隐藏层状态
		shape = [self.batch, self.num_hidden, self.height, self.width]
		return torch.zeros(shape, dtype=torch.float32).cuda()

	def forward(self, x, h_t, c_t, batch_size):
		# h c [batch, num_hidden, in_height, in_width]
		self.batch = batch_size	
		if h_t is None:
			h_t = self.init_state()
		if c_t is None:
			c_t = self.init_state()

		x = x[:self.batch]
		h_t = h_t[:self.batch]
		c_t = c_t[:self.batch]
		
		# 1
		h_concat = self.h_t(h_t)
		
		if self.layer_norm:
			h_concat = self.bn_h_concat(h_concat)
		i_h, g_h, f_h, o_h = torch.split(h_concat, self.num_hidden, 1)
		
		# 2 变量 可训练
		ct_activation = torch.mul(c_t.repeat([1,2,1,1]), self.ct_weight)
		i_c, f_c = torch.split(ct_activation, self.num_hidden, 1)

		i_ = i_h + i_c
		f_ = f_h + f_c
		g_ = g_h
		o_ = o_h

		if x is not None:
			# 3 x
			x_concat = self.x(x)
			
			if self.layer_norm:
				x_concat = self.bn_x_concat(x_concat)
			i_x, g_x, f_x, o_x = torch.split(x_concat, self.num_hidden, 1)

			i_ = i_ + i_x
			f_ = f_ + f_x
			g_ = g_ + g_x
			o_ = o_ + o_x

		i_ = torch.sigmoid(i_)
		f_ = torch.sigmoid(f_ + self._forget_bias)
		c_new = f_ * c_t + i_ * torch.tanh(g_)

		# 4 变量 可训练
		o_c = torch.mul(c_new, self.oc_weight)

		h_new = torch.sigmoid(o_ + o_c) * torch.tanh(c_new)

		return h_new, c_new


class MIM_SpatioTemporalLSTMCell(nn.Module): 
	def __init__(self, layer_name, filter_size, num_hidden_in, num_hidden,
				 seq_shape, x_shape_in, tln=False, initializer=None):
		super(MIM_SpatioTemporalLSTMCell, self).__init__()
		
		"""Initialize the basic Conv LSTM cell.
		Args:
			layer_name: layer names for different convlstm layers.
			filter_size: int tuple thats the height and width of the filter.
			num_hidden: number of units in output tensor.
			forget_bias: float, The bias added to forget gates (see above).
			tln: whether to apply tensor layer normalization
		"""
		self.layer_name = layer_name 
		self.filter_size = filter_size 
		self.num_hidden_in = num_hidden_in 
		self.num_hidden = num_hidden 
		self.batch = seq_shape[0] 
		self.height = seq_shape[3] 
		self.width = seq_shape[4] 
		self.seq_shape = seq_shape
		self.x_shape_in = x_shape_in 
		self.layer_norm = tln 
		self._forget_bias = 1.0 
			
		# h
		self.t_cc = nn.Conv2d(self.num_hidden_in,
				self.num_hidden*4,
				self.filter_size, 1, padding = 2 
				)
				
		# m
		self.s_cc = nn.Conv2d(self.num_hidden_in,
				self.num_hidden*4,  
				self.filter_size, 1, padding = 2 
				)
				
		# x
		self.x_cc = nn.Conv2d(self.x_shape_in,
				self.num_hidden*4, 
				self.filter_size, 1, padding = 2 
				)
		
		# c 
		self.c_cc = nn.Conv2d(self.num_hidden*2,
				self.num_hidden,  
				1, 1, padding = 0 
				)
				
		self.bn_t_cc = nn.BatchNorm2d(self.num_hidden*4)
		self.bn_s_cc = nn.BatchNorm2d(self.num_hidden*4)
		self.bn_x_cc = nn.BatchNorm2d(self.num_hidden*4)
				
	def init_state(self): # 初始化lstm 隐藏层状态
		return torch.zeros((self.batch, self.num_hidden, self.height, self.width),
						dtype=torch.float32).cuda()

	def forward(self, x, h, c, m, batch_size):
		self.batch = batch_size
		# 初始化隐藏层 记忆 空间
		if h is None:
			h = self.init_state()
		if c is None:
			c = self.init_state()
		if m is None:
			m = self.init_state()
		
		x = x[:self.batch]
		h = h[:self.batch]
		c = c[:self.batch]
		m = m[:self.batch]		
		# 计算网络输出
		t_cc = self.t_cc(h)
		s_cc = self.s_cc(m)
		x_cc = self.x_cc(x)
		
		if self.layer_norm:
			# 计算均值 标准差 归一化
			t_cc = self.bn_t_cc(t_cc)
			s_cc = self.bn_s_cc(s_cc)
			x_cc = self.bn_x_cc(x_cc)
		
		# 在第3维度上切分为4份 因为隐藏层是4*num_hidden 
		i_s, g_s, f_s, o_s = torch.split(s_cc, self.num_hidden, 1) # [batch, num_hidden, in_height, in_width]
		i_t, g_t, f_t, o_t = torch.split(t_cc, self.num_hidden, 1)
		i_x, g_x, f_x, o_x = torch.split(x_cc, self.num_hidden, 1)

		i = torch.sigmoid(i_x + i_t)
		i_ = torch.sigmoid(i_x + i_s)
		g = torch.tanh(g_x + g_t)
		g_ = torch.tanh(g_x + g_s)
		f = torch.sigmoid(f_x + f_t + self._forget_bias)
		f_ = torch.sigmoid(f_x + f_s + self._forget_bias)
		o = torch.sigmoid(o_x + o_t + o_s)
		new_m = f_ * m + i_ * g_
		new_c = f * c + i * g
		cell = torch.cat((new_c, new_m),1) # [batch, 2*num_hidden, in_height, in_width]
		
		tmp_cell = self.c_cc(cell)
		new_h = o * torch.tanh(tmp_cell)

		# print(new_h.shape, new_c.shape, new_m.shape)f
		return new_h, new_c, new_m


# Cite from https://github.com/metrofun/E3D-LSTM
class E3DLSTMCell(nn.Module):
    def __init__(self, input_shape, hidden_size, kernel_size):
        super().__init__()

        in_channels = input_shape[0]
        self._input_shape = input_shape
        self._hidden_size = hidden_size

        # memory gates: input, cell(input modulation), forget
        self.weight_xi = ConvDeconv3d(in_channels, hidden_size, kernel_size)
        self.weight_hi = ConvDeconv3d(hidden_size, hidden_size, kernel_size, bias=False)

        self.weight_xg = copy.deepcopy(self.weight_xi)
        self.weight_hg = copy.deepcopy(self.weight_hi)

        self.weight_xr = copy.deepcopy(self.weight_xi)
        self.weight_hr = copy.deepcopy(self.weight_hi)

        memory_shape = list(input_shape)
        memory_shape[0] = hidden_size

        self.layer_norm = nn.LayerNorm(memory_shape)

        # for spatiotemporal memory
        self.weight_xi_prime = copy.deepcopy(self.weight_xi)
        self.weight_mi_prime = copy.deepcopy(self.weight_hi)

        self.weight_xg_prime = copy.deepcopy(self.weight_xi)
        self.weight_mg_prime = copy.deepcopy(self.weight_hi)

        self.weight_xf_prime = copy.deepcopy(self.weight_xi)
        self.weight_mf_prime = copy.deepcopy(self.weight_hi)

        self.weight_xo = copy.deepcopy(self.weight_xi)
        self.weight_ho = copy.deepcopy(self.weight_hi)
        self.weight_co = copy.deepcopy(self.weight_hi)
        self.weight_mo = copy.deepcopy(self.weight_hi)

        self.weight_111 = nn.Conv3d(hidden_size + hidden_size, hidden_size, 1)

    def self_attention(self, r, c_history):
        batch_size = r.size(0)
        channels = r.size(1)
        r_flatten = r.view(batch_size, -1, channels)
        # BxtaoTHWxC
        c_history_flatten = c_history.view(batch_size, -1, channels)

        # Attention mechanism
        # BxTHWxC x BxtaoTHWxC' = B x THW x taoTHW
        scores = torch.einsum("bxc,byc->bxy", r_flatten, c_history_flatten)
        attention = F.softmax(scores, dim=2)

        return torch.einsum("bxy,byc->bxc", attention, c_history_flatten).view(*r.shape)

    def self_attention_fast(self, r, c_history):
        # Scaled Dot-Product but for tensors
        # instead of dot-product we do matrix contraction on twh dimensions
        scaling_factor = 1 / (reduce(operator.mul, r.shape[-3:], 1) ** 0.5)
        scores = torch.einsum("bctwh,lbctwh->bl", r, c_history) * scaling_factor

        attention = F.softmax(scores, dim=0)
        return torch.einsum("bl,lbctwh->bctwh", attention, c_history)

    def forward(self, x, c_history, m, h):
        # Normalized shape for LayerNorm is CxT×H×W
        normalized_shape = list(h.shape[-3:])

        def LR(input):
            return F.layer_norm(input, normalized_shape)

        # R is CxT×H×W
        r = torch.sigmoid(LR(self.weight_xr(x) + self.weight_hr(h)))
        i = torch.sigmoid(LR(self.weight_xi(x) + self.weight_hi(h)))
        g = torch.tanh(LR(self.weight_xg(x) + self.weight_hg(h)))

        recall = self.self_attention_fast(r, c_history)

        c = i * g + self.layer_norm(c_history[-1] + recall)

        i_prime = torch.sigmoid(LR(self.weight_xi_prime(x) + self.weight_mi_prime(m)))
        g_prime = torch.tanh(LR(self.weight_xg_prime(x) + self.weight_mg_prime(m)))
        f_prime = torch.sigmoid(LR(self.weight_xf_prime(x) + self.weight_mf_prime(m)))

        m = i_prime * g_prime + f_prime * m
        o = torch.sigmoid(
            LR(
                self.weight_xo(x)
                + self.weight_ho(h)
                + self.weight_co(c)
                + self.weight_mo(m)
            )
        )
        h = o * torch.tanh(self.weight_111(torch.cat([c, m], dim=1)))

        # TODO is it correct FIFO?
        c_history = torch.cat([c_history[1:], c[None, :]], dim=0)

        return (c_history, m, h)

    def init_hidden(self, batch_size, tau, device=None):
        memory_shape = list(self._input_shape)
        memory_shape[0] = self._hidden_size
        c_history = torch.zeros(tau, batch_size, *memory_shape, device=device)
        m = torch.zeros(batch_size, *memory_shape, device=device)
        h = torch.zeros(batch_size, *memory_shape, device=device)

        return (c_history, m, h)


class ConvDeconv3d(nn.Module):
    def __init__(self, in_channels, out_channels, *vargs, **kwargs):
        super().__init__()

        self.conv3d = nn.Conv3d(in_channels, out_channels, *vargs, **kwargs)
        # self.conv_transpose3d = nn.ConvTranspose3d(out_channels, out_channels, *vargs, **kwargs)

    def forward(self, input):
        # print(self.conv3d(input).shape, input.shape)
        # return self.conv_transpose3d(self.conv3d(input))
        return F.interpolate(self.conv3d(input), size=input.shape[-3:], mode="nearest")