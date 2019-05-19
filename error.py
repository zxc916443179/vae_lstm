
class ModeNotDefinedError(Exception):
    def __init__(self, mode, message='given mode not found, expect (train or finetune) but get '):
        self.mode = message + mode
        self.message = message + self.mode