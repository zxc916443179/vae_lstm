
class ModeNotDefinedError(Exception):
    def __init__(self, mode, message='given mode not found, expect (train or finetune) but get '):
        self.mode = mode
        self.message = message + self.mode