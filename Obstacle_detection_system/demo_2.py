
class A:
    def __init__(self):
        pass

    def get_num(self):
        return  self.num

    @property
    def num(self):
        pass


class B(A):#A为父类 B为子类
    def __init__(self):
        super().__init__()

    @property
    def num(self):
        return 233

if __name__ == '__main__':
    b = B()
    print(b.get_num())