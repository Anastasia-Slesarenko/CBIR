import uvicorn
import sys

sys.path.append("../")

from lib.app import app

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")
