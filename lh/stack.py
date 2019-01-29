class stack:
	def __init__(self):
		self.s_list = []
		self._length = 0
		self.top = None
		self.bottom = None

	def pop(self):
		return 	self.s_list.pop()


	def push(self, data):
		self.s_list.append(data)
		self.top = data

	def is_empty(self):
		if len(self.s_list) == 0:
			return True
		else:
			return False

	def length(self):
		return len(self.s_list)