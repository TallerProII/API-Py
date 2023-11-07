from fastapi import FastAPI
from pydantic import BaseModel
import prediccion

app = FastAPI()
#http://127.0.0.0:8000

class Imagen(BaseModel):
        img_name:str
        cod_base64: str

@app.get("/")
def index():
        return {"mensaje": "Hola"}

@app.get("/Imagen/{id}")
def mostrar_img(id: int):
        return {"data":id}

@app.get("/Prueba/{id}")
def dt(data: str):
        return prediccion.funcion(data)

@app.post("/Img_base64")
def insertar_img(imagen: Imagen):
        return {"cod_base64": imagen}
