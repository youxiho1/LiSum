from transformers.trainer import Trainer

class MyTrainerLite(Trainer):
    def _load_best_model(self):
        self.best_model_path = self.state.best_model_checkpoint
        super()._load_best_model()
    
    def get_best_model_path(self):
        return self.best_model_path