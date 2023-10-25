class vector:
    def __init__ (self, size, isCol=True, fill=0, initVals=None):
        self.isCol = isCol
        self.vals = []
        self.size = size
        if initVals == None:
            for i in range(size):
                self.vals.append(fill)
        if initVals != None:
            for i in range(size):
                self.vals.append(initVals[i%len(initVals)])

    def __str__ (self):
        ret = "["
        ending = (",\n" if self.isCol else ", ")
        for v in self.vals:
            ret += str(v) + ending
        ret = ret[:-2]
        ret += "]"
        return ret

    def transpose(self):
        return vector (self.size, not self.isCol, initVals = self.vals)

    def dot(self, other):
        if self.size != other.size:
            raise Exception("vectors must be the same size for dot product.")
        if self.isCol or (not other.isCol):
            raise Exception("vectors need to be a row and a column (respectively) for dot product.")
        ret = 0
        tempVector = self * other.transpose()
        for i in tempVector:
            ret += i
        return ret


    def __checkCompatibility(self, other):
      if self.size != other.size:
        raise Exception("vectors must be the same size for math operations.")
      if self.isCol != other.isCol:
        raise Exception("vectors must both be the same axis for math operations.")

    def __add__(self, other):
        retVals = []
        if isinstance(other, vector):
            self.__checkCompatibility(other)
            for i in range(self.size):
                retVals.append(self.vals[i] + other.vals[i])
        else:
            for i in range(self.size):
                retVals.append(self.vals[i] + other)
        return vector(self.size, self.isCol, initVals = retVals)
    
    def __sub__(self, other):
        return self + other * (-1)

    def __mul__(self, other):
        retVals = []
        if isinstance(other, vector):
            self.__checkCompatibility(other)
            for i in range(self.size):
                retVals.append(self.vals[i] * other.vals[i])
        else:
            for i in range(self.size):
                retVals.append(self.vals[i] * other)
        return vector(self.size, self.isCol, initVals = retVals)

    def __truediv__(self, other):
        retVals = []
        if isinstance(other, vector):
            self.__checkCompatibility(other)
            for i in range(self.size):
                retVals.append(self.vals[i] / other.vals[i])
        else:
            for i in range(self.size):
                retVals.append(self.vals[i] / other)
        return vector(self.size, self.isCol, initVals = retVals)

    def __getitem__(self, key):
        return self.vals[key]

    def __setitem__(self, key, value):
        self.vals[key] = value

    def __len__(self):
        return len(self.vals)

    def __radd__(self, other):
        return self + other

    def __rsub__(self, other):
        return self * (-1) + other

    def __rtruediv__(self, other):
        return vector(self.size, self.isCol, fill=other) / self
    
    def __rmul__(self, other):
        return self * other

    def __lt__(self, other):
        retVals = []
        for i in range(self.size):
            retVals.append( 1 if self[i] < other else 0)
        return vector(self.size, self.isCol, initVals = retVals)

    def __le__(self, other):
        retVals = []
        for i in range(self.size):
            retVals.append( 1 if self[i] <= other else 0)
        return vector(self.size, self.isCol, initVals = retVals)

    def __eq__(self, other):
        retVals = []
        for i in range(self.size):
            retVals.append( 1 if self[i] == other else 0)
        return vector(self.size, self.isCol, initVals = retVals)

    def __ne__(self, other):
        retVals = []
        for i in range(self.size):
            retVals.append( 1 if self[i] != other else 0)
        return vector(self.size, self.isCol, initVals = retVals)

    def __gt__(self, other):
        retVals = []
        for i in range(self.size):
            retVals.append( 1 if self[i] > other else 0)
        return vector(self.size, self.isCol, initVals = retVals)

    def __ge__(self, other):
        retVals = []
        for i in range(self.size):
            retVals.append( 1 if self[i] >= other else 0)
        return vector(self.size, self.isCol, initVals = retVals)
