from src.constants.constants import HOST, PORT
from src.api.endpoints import router
from fastapi import FastAPI
import uvicorn

app = FastAPI()

app.include_router(router)

if __name__ == "__main__":
    uvicorn.run(app, host=HOST, port=PORT)
