
class CPPBase:
    def __init__(self):
        self.c_class = None
        self.c_instance = None
    
    def init_c_instance(self, **init_args):
        if self.c_class is None:
            raise RuntimeError("c_class is None. ")
        
        self.c_instance = self.c_class(**init_args)
        