from src.webapp.src.app import app
from src.webapp.src.model_manager import model_manager

if __name__ == '__main__':
    model_manager.load_models()
    app.run()
