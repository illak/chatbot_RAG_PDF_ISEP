# Chatbot con base de conocimiento en PDFs

La aplicación permite cargar PDFs y "chatear" con ellos.

Se usó como base [este desarrollo](https://github.com/kaifcoder/gemini_multipdf_chat), créditos al desarrollador: *[kaifcoder](https://github.com/kaifcoder)*.

## Para arrancar

El requisito principal es la [clave API de Google](https://ai.google.dev/) generada. Una vez obtenida, agregar al archivo `.env`.
Puede guiarse con el template `.env.example`.

```.env
GOOGLE_API_KEY=your_api_key_here
```
Luego, con docker instalado, se puede levantar la aplicación usando el siguiente comando:

```bash
docker compose up --build
```

La aplicación estará disponible en: <http://localhost:8501>.