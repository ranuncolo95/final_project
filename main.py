from transformers import pipeline
from ultralytics import YOLO
from fastapi import FastAPI, Request, Form, File, UploadFile, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from PIL import Image

app = FastAPI()

templates = Jinja2Templates(directory="templates")

# PRIMA PARTE


@app.get("/", response_class=HTMLResponse)
def home(request: Request):
    return templates.TemplateResponse(request=request, name="index.html")


@app.post("/", response_class=HTMLResponse)
def services(request: Request, 
             input_text: str = Form(...), 
             from_lang: str = Form(...), 
             to_lang: str = Form(...)):
    try: 
        generator = pipeline(f"translation_{from_lang}_to_{to_lang}")
        translation = generator(input_text)
        return templates.TemplateResponse(request=request, 
                                        name="index.html", 
                                        context={'translation': translation[0]["translation_text"]})
    except:
        return templates.TemplateResponse(request=request, 
                                        name="index.html", 
                                        context={'translation': "Scrivi qualcosa in input"})


# SECONDA PARTE



@app.get("/image", response_class=HTMLResponse)
def image(request: Request):
    return templates.TemplateResponse(request=request, name="image.html")


@app.post("/image", response_class=HTMLResponse)
def image_analisy(request: Request, obj: int = Form(...), input_image: UploadFile = File(...)):
    model = YOLO("yolov8s")

    analize_image = model(input_image.filename, show=False)

    oggetti_trovati = analize_image[0].boxes.cls.tolist()
    result = oggetti_trovati.count(obj)

    print(result)
    return templates.TemplateResponse(
        request=request, 
        name="image.html", 
        context={"obj_count": f"L'oggetto analizzato è presente {result} volte nell'immagine."}
        )

