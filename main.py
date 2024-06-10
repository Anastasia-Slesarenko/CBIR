import uvicorn
from lib.app import app
from lib.settings import APP_PORT


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=APP_PORT, log_level="info")
