class MarginLoss(nn.Module):
	def __init__(self, m_pos, m_neg, lambda_):
		super(MarginLoss, self).__init__()
		self.m_pos = m_pos
		self.m_neg = m_neg
		self.lambda_ = lambda_

	def forward(self, y_hat, batch_y, size_average=True):
		t = torch.ones(y_hat.size())
		if batch_y.is_cuda:
			t = t.cuda()
		# t = t.scatter_(1, batch_y.data.view(-1, 1), 1)
		# batch_y = Variable(t)
		losses = batch_y.float() * F.relu(self.m_pos*t - y_hat).pow(2) + \
				self.lambda_ * (1. - batch_y.float()) * F.relu(y_hat - self.m_neg*t).pow(2)
		return losses.mean() if size_average else losses.sum()
